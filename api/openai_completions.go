package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
	openaissestream "github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/shared"

	"github.com/yoneyy/aimix"
)

// maxCompletionsToolCallIDLen is the longest tool call id OpenAI accepts.
const maxCompletionsToolCallIDLen = 40

// completionsReasoningFields are the delta fields OpenAI-compatible servers
// stream reasoning in, in order of preference. The field a thinking block came
// from is kept as its signature so it can be sent back the same way.
var completionsReasoningFields = []string{
	"reasoning_content",
	"reasoning",
	"reasoning_text",
}

// OpenAICompletionsOptions are the options of the openai-completions Api.
type OpenAICompletionsOptions struct {
	aimix.StreamOptions
	// ReasoningEffort is the reasoning_effort wire value. Empty leaves the
	// model default.
	ReasoningEffort string
}

type openAICompletions struct{}

// OpenAICompletions returns the openai-completions implementation. Besides
// OpenAI it works with OpenAI-compatible servers through Model.BaseURL.
func OpenAICompletions() aimix.ApiStreamer {
	return openAICompletions{}
}

func (openAICompletions) Stream(ctx context.Context, req aimix.StreamRequest) (*aimix.AssistantMessageEventStream, error) {
	return StreamOpenAICompletions(ctx, req.Model, req.Context, OpenAICompletionsOptions{
		StreamOptions: req.Options,
	})
}

func (openAICompletions) StreamSimple(ctx context.Context, req aimix.SimpleStreamRequest) (*aimix.AssistantMessageEventStream, error) {
	return StreamSimpleOpenAICompletions(ctx, req.Model, req.Context, req.Options)
}

// StreamOpenAICompletions streams one assistant turn from the Chat Completions API.
func StreamOpenAICompletions(ctx context.Context, model aimix.Model, tc aimix.TranscriptContext, opts OpenAICompletionsOptions) (*aimix.AssistantMessageEventStream, error) {
	return streamCompletions(ctx, model, tc, opts, "")
}

// StreamSimpleOpenAICompletions maps the unified thinking level to reasoning_effort.
func StreamSimpleOpenAICompletions(ctx context.Context, model aimix.Model, tc aimix.TranscriptContext, opts aimix.SimpleStreamOptions) (*aimix.AssistantMessageEventStream, error) {
	level := effectiveReasoning(model, opts.Reasoning)
	o := OpenAICompletionsOptions{
		StreamOptions: opts.StreamOptions,
	}
	o.MaxTokens = baseMaxTokens(model, tc.Messages, opts.StreamOptions)
	o.ReasoningEffort = openAIEffort(model, level)
	return streamCompletions(ctx, model, tc, o, level)
}

func streamCompletions(ctx context.Context, model aimix.Model, tc aimix.TranscriptContext, opts OpenAICompletionsOptions, level aimix.ThinkingLevel) (*aimix.AssistantMessageEventStream, error) {
	if opts.APIKey == "" {
		return nil, errMissingAPIKey
	}
	params, err := buildCompletionsParams(model, tc, opts)
	if err != nil {
		return nil, err
	}
	service := openai.NewChatCompletionService(openAIRequestOptions(model, opts.StreamOptions)...)

	w := newStreamWriter(ctx, model)
	w.out.ThinkingLevel = level
	return w.run(func() error {
		var resp *http.Response
		stream := service.NewStreaming(ctx, params, openaioption.WithResponseInto(&resp))
		defer stream.Close()
		if err := stream.Err(); err != nil {
			return err
		}
		if err := checkEventStream(resp); err != nil {
			return err
		}
		w.start()
		return processCompletionsStream(w, model, stream)
	}), nil
}

func buildCompletionsParams(model aimix.Model, tc aimix.TranscriptContext, opts OpenAICompletionsOptions) (openai.ChatCompletionNewParams, error) {
	system, rest := splitSystem(aimix.CollapseSystemMessages(tc).Messages)
	messages := transformMessages(rest, model, normalizeCompletionsToolCallID)

	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(model.ID),
		Messages: convertCompletionsMessages(model, system, messages),
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		},
	}

	// store and prompt caching parameters are OpenAI-only; compatible servers may
	// reject them.
	baseURL := model.BaseURL
	if baseURL == "" {
		baseURL = defaultOpenAIBaseURL
	}
	if strings.Contains(baseURL, "api.openai.com") {
		params.Store = openai.Bool(false)
		if opts.CacheRetention != aimix.CacheRetentionNone && opts.SessionID != "" {
			key := opts.SessionID
			if len(key) > maxPromptCacheKeyLen {
				key = key[:maxPromptCacheKeyLen]
			}
			params.PromptCacheKey = openai.String(key)
		}
		if opts.CacheRetention == aimix.CacheRetentionLong {
			params.PromptCacheRetention = openai.ChatCompletionNewParamsPromptCacheRetention24h
		}
	}

	if opts.MaxTokens > 0 {
		if model.Compat.LegacyMaxTokens {
			params.MaxTokens = openai.Int(int64(opts.MaxTokens))
		} else {
			params.MaxCompletionTokens = openai.Int(int64(opts.MaxTokens))
		}
	}
	if opts.Temperature != nil && !model.Compat.NoTemperature {
		params.Temperature = openai.Float(*opts.Temperature)
	}

	if len(system.ToolsAdded) > 0 {
		tools := make([]openai.ChatCompletionToolUnionParam, 0, len(system.ToolsAdded))
		for _, t := range system.ToolsAdded {
			schema, err := toolSchemaMap(t)
			if err != nil {
				return params, err
			}
			fn := shared.FunctionDefinitionParam{
				Name:       t.Name,
				Parameters: schema,
			}
			if t.Description != "" {
				fn.Description = openai.String(t.Description)
			}
			if t.Strict {
				fn.Strict = openai.Bool(true)
			}
			tools = append(tools, openai.ChatCompletionFunctionTool(fn))
		}
		params.Tools = tools
	} else if hasToolHistory(messages) {
		// Some servers reject tool calls in the history when no tools are declared.
		params.Tools = []openai.ChatCompletionToolUnionParam{}
	}
	if choice := string(opts.ToolChoice); choice != "" {
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String(choice),
		}
	}

	if model.Reasoning && opts.ReasoningEffort != "" && !model.Compat.NoReasoningEffort {
		params.ReasoningEffort = shared.ReasoningEffort(opts.ReasoningEffort)
	}
	return params, nil
}

func hasToolHistory(messages []aimix.Message) bool {
	for _, m := range messages {
		switch v := m.(type) {
		case aimix.ToolResultMessage:
			return true
		case aimix.AssistantMessage:
			for _, c := range v.Content {
				if _, ok := c.(aimix.ToolCallContent); ok {
					return true
				}
			}
		}
	}
	return false
}

// normalizeCompletionsToolCallID makes a foreign tool call id fit Chat
// Completions: "call_id|item_id" ids from the Responses API keep both parts so
// they stay unique, and ids over 40 bytes are shortened with a hash.
func normalizeCompletionsToolCallID(id string) string {
	if callID, itemID, ok := strings.Cut(id, "|"); ok {
		callID = sanitizeID(callID, len(callID))
		itemID = sanitizeID(itemID, len(itemID))
		combined := callID
		if itemID != "" {
			combined += "_" + itemID
		}
		if len(combined) <= maxCompletionsToolCallIDLen {
			return combined
		}
		hash := shortHash(id)
		if len(hash) > 8 {
			hash = hash[:8]
		}
		prefixLen := max(1, maxCompletionsToolCallIDLen-len(hash)-1)
		if len(callID) > prefixLen {
			callID = callID[:prefixLen]
		}
		return callID + "_" + hash
	}
	if len(id) > maxCompletionsToolCallIDLen {
		return id[:maxCompletionsToolCallIDLen]
	}
	return id
}

func convertCompletionsMessages(model aimix.Model, system aimix.SystemMessage, messages []aimix.Message) []openai.ChatCompletionMessageParamUnion {
	var params []openai.ChatCompletionMessageParamUnion
	if text := aimix.TextOf(system.Content); text != "" {
		params = append(params, openai.SystemMessage(text))
	}

	for i := 0; i < len(messages); i++ {
		switch m := messages[i].(type) {
		case aimix.UserMessage:
			if !hasImages(m.Content) {
				if text := strings.Join(textBlocks(m.Content), "\n"); text != "" {
					params = append(params, openai.UserMessage(text))
				}
				continue
			}
			var parts []openai.ChatCompletionContentPartUnionParam
			for _, c := range m.Content {
				switch b := c.(type) {
				case aimix.TextContent:
					parts = append(parts, openai.TextContentPart(b.Text))
				case aimix.ImageContent:
					parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
						URL: dataURL(b),
					}))
				}
			}
			params = append(params, openai.UserMessage(parts))

		case aimix.AssistantMessage:
			if p, ok := completionsAssistantMessage(m); ok {
				params = append(params, p)
			}

		case aimix.ToolResultMessage:
			// Tool messages only carry text: images of consecutive tool results
			// follow them in one user message.
			var images []openai.ChatCompletionContentPartUnionParam
			j := i
			for ; j < len(messages); j++ {
				tr, ok := messages[j].(aimix.ToolResultMessage)
				if !ok {
					break
				}
				text := strings.Join(textBlocks(tr.Content), "\n")
				withImages := hasImages(tr.Content)
				switch {
				case strings.TrimSpace(text) != "":
				case withImages:
					text = "(see attached image)"
				default:
					text = "(no tool output)"
				}
				params = append(params, openai.ToolMessage(text, tr.ToolCallID))
				if withImages && model.SupportsImages() {
					for _, c := range tr.Content {
						if img, ok := c.(aimix.ImageContent); ok {
							images = append(images, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
								URL: dataURL(img),
							}))
						}
					}
				}
			}
			i = j - 1
			if len(images) > 0 {
				parts := append([]openai.ChatCompletionContentPartUnionParam{
					openai.TextContentPart("Attached image(s) from tool result:"),
				}, images...)
				params = append(params, openai.UserMessage(parts))
			}
		}
	}
	return params
}

func completionsAssistantMessage(m aimix.AssistantMessage) (openai.ChatCompletionMessageParamUnion, bool) {
	var (
		text      strings.Builder
		thinking  []string
		signature string
		calls     []openai.ChatCompletionMessageToolCallUnionParam
	)
	for _, c := range m.Content {
		switch b := c.(type) {
		case aimix.TextContent:
			text.WriteString(b.Text)
		case aimix.ThinkingContent:
			if strings.TrimSpace(b.Thinking) == "" {
				continue
			}
			thinking = append(thinking, b.Thinking)
			if signature == "" {
				signature = b.ThinkingSignature
			}
		case aimix.ToolCallContent:
			args := string(b.Arguments)
			if args == "" {
				args = "{}"
			}
			calls = append(calls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: b.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      b.Name,
						Arguments: args,
					},
				},
			})
		}
	}

	p := openai.ChatCompletionAssistantMessageParam{
		ToolCalls: calls,
	}
	if text.Len() > 0 {
		p.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
			OfString: openai.String(text.String()),
		}
	}
	if text.Len() == 0 && len(calls) == 0 {
		return openai.ChatCompletionMessageParamUnion{}, false
	}
	// Thinking goes back in the field it was streamed in. transformMessages has
	// already dropped thinking from other models.
	if len(thinking) > 0 && slices.Contains(completionsReasoningFields, signature) {
		p.SetExtraFields(map[string]any{
			signature: strings.Join(thinking, "\n"),
		})
	}
	return openai.ChatCompletionMessageParamUnion{
		OfAssistant: &p,
	}, true
}

func processCompletionsStream(w *streamWriter, model aimix.Model, stream *openaissestream.Stream[openai.ChatCompletionChunk]) error {
	var (
		textIdx     = -1
		thinkingIdx = -1
		toolByIndex = make(map[int64]int)
		toolByID    = make(map[string]int)
		order       []int // content indices in creation order
		finished    bool
		stopErr     error
	)
	for stream.Next() {
		chunk := stream.Current()
		if w.out.ResponseID == "" {
			w.out.ResponseID = chunk.ID
		}
		if chunk.Model != "" && chunk.Model != model.ID && w.out.ResponseModel == "" {
			w.out.ResponseModel = chunk.Model
		}
		if chunk.JSON.Usage.Valid() {
			w.out.Usage = completionsUsage(chunk.Usage)
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		if choice.FinishReason != "" {
			finished = true
			w.out.RawStopReason = choice.FinishReason
			w.out.StopReason, stopErr = mapCompletionsFinishReason(choice.FinishReason)
		}

		delta := choice.Delta
		for _, field := range completionsReasoningFields {
			f, ok := delta.JSON.ExtraFields[field]
			if !ok {
				continue
			}
			var reasoning string
			if json.Unmarshal([]byte(f.Raw()), &reasoning) != nil || reasoning == "" {
				continue
			}
			if thinkingIdx < 0 {
				thinkingIdx = w.startThinking()
				order = append(order, thinkingIdx)
				w.setThinking(thinkingIdx, func(c *aimix.ThinkingContent) {
					c.ThinkingSignature = field
				})
			}
			w.appendThinking(thinkingIdx, reasoning)
			break
		}

		if delta.Content != "" {
			if textIdx < 0 {
				textIdx = w.startText()
				order = append(order, textIdx)
			}
			w.appendText(textIdx, delta.Content)
		}

		for _, tc := range delta.ToolCalls {
			i, ok := toolByIndex[tc.Index]
			if !ok && tc.ID != "" {
				i, ok = toolByID[tc.ID]
			}
			if !ok {
				i = w.startToolCall(tc.ID, tc.Function.Name)
				order = append(order, i)
				toolByIndex[tc.Index] = i
			}
			if tc.ID != "" {
				toolByID[tc.ID] = i
			}
			w.updateToolCall(i, func(c *aimix.ToolCallContent) {
				if c.ID == "" {
					c.ID = tc.ID
				}
				if c.Name == "" {
					c.Name = tc.Function.Name
				}
			})
			w.appendToolArgs(i, tc.Function.Arguments)
		}
	}
	if err := stream.Err(); err != nil {
		return err
	}

	// Chat Completions has no block end events: blocks end with the stream.
	for _, i := range order {
		switch w.out.Content[i].(type) {
		case aimix.TextContent:
			w.endText(i)
		case aimix.ThinkingContent:
			w.endThinking(i)
		case aimix.ToolCallContent:
			w.endToolCall(i)
		}
	}
	if !finished {
		return errors.New("aimix: Chat Completions stream ended without finish_reason")
	}
	return stopErr
}

func mapCompletionsFinishReason(reason string) (aimix.StopReason, error) {
	switch reason {
	case "stop", "end":
		return aimix.StopReasonStop, nil
	case "length":
		return aimix.StopReasonLength, nil
	case "tool_calls", "function_call":
		return aimix.StopReasonToolUse, nil
	}
	return aimix.StopReasonError, fmt.Errorf("provider finish_reason: %s", reason)
}

func completionsUsage(u openai.CompletionUsage) aimix.Usage {
	cacheRead := int(u.PromptTokensDetails.CachedTokens)
	if cacheRead == 0 {
		// DeepSeek reports cache hits outside prompt_tokens_details.
		if f, ok := u.JSON.ExtraFields["prompt_cache_hit_tokens"]; ok {
			_ = json.Unmarshal([]byte(f.Raw()), &cacheRead)
		}
	}
	cacheWrite := int(u.PromptTokensDetails.CacheWriteTokens)
	usage := aimix.Usage{
		Input:      max(0, int(u.PromptTokens)-cacheRead-cacheWrite),
		Output:     int(u.CompletionTokens),
		CacheRead:  cacheRead,
		CacheWrite: cacheWrite,
		Reasoning:  int(u.CompletionTokensDetails.ReasoningTokens),
	}
	usage.TotalTokens = usage.Input + usage.Output + usage.CacheRead + usage.CacheWrite
	return usage
}

package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	anthropicparam "github.com/anthropics/anthropic-sdk-go/packages/param"
	anthropicssestream "github.com/anthropics/anthropic-sdk-go/packages/ssestream"

	"github.com/yoneyy/aimix"
)

const (
	// defaultAnthropicMaxTokens is sent when neither the caller nor the model sets
	// a cap; the Messages API requires max_tokens.
	defaultAnthropicMaxTokens = 4096
	// minAnthropicThinkingBudget is the smallest budget_tokens the API accepts.
	minAnthropicThinkingBudget = 1024
	interleavedThinkingBeta    = "interleaved-thinking-2025-05-14"
	errorToolImagePlaceholder  = "(image attached after the tool results)"
)

// AnthropicOptions are the options of the anthropic-messages Api.
type AnthropicOptions struct {
	aimix.StreamOptions
	// ThinkingEnabled turns thinking on or off. Nil leaves the model default.
	ThinkingEnabled *bool
	// ThinkingBudgetTokens is the budget of budget-based thinking. 0 uses 1024.
	ThinkingBudgetTokens int
	// Effort is the output_config effort ("low", "medium", "high", "xhigh", "max")
	// of models with Compat.AdaptiveThinking.
	Effort string
}

type anthropicMessages struct{}

// AnthropicMessages returns the anthropic-messages implementation.
func AnthropicMessages() aimix.ApiStreamer {
	return anthropicMessages{}
}

func (anthropicMessages) Stream(ctx context.Context, req aimix.StreamRequest) (*aimix.AssistantMessageEventStream, error) {
	return StreamAnthropicMessages(ctx, req.Model, req.Context, AnthropicOptions{
		StreamOptions: req.Options,
	})
}

func (anthropicMessages) StreamSimple(ctx context.Context, req aimix.SimpleStreamRequest) (*aimix.AssistantMessageEventStream, error) {
	return StreamSimpleAnthropicMessages(ctx, req.Model, req.Context, req.Options)
}

// StreamAnthropicMessages streams one assistant turn from the Anthropic Messages API.
func StreamAnthropicMessages(ctx context.Context, model aimix.Model, tc aimix.TranscriptContext, opts AnthropicOptions) (*aimix.AssistantMessageEventStream, error) {
	return streamAnthropic(ctx, model, tc, opts, "")
}

// StreamSimpleAnthropicMessages maps the unified thinking level to Anthropic
// options: adaptive thinking with an effort for models with
// Compat.AdaptiveThinking, a token budget for the others.
func StreamSimpleAnthropicMessages(ctx context.Context, model aimix.Model, tc aimix.TranscriptContext, opts aimix.SimpleStreamOptions) (*aimix.AssistantMessageEventStream, error) {
	level := effectiveReasoning(model, opts.Reasoning)
	o := AnthropicOptions{
		StreamOptions: opts.StreamOptions,
	}
	o.MaxTokens = baseMaxTokens(model, tc.Messages, opts.StreamOptions)

	enabled := level != aimix.ThinkingOff
	o.ThinkingEnabled = &enabled
	switch {
	case !enabled:
	case model.Compat.AdaptiveThinking:
		o.Effort = anthropicEffort(model, level)
	default:
		maxTokens, budget := adjustMaxTokensForThinking(o.MaxTokens, model.MaxTokens, level, opts.ThinkingBudgets)
		if maxTokens <= 0 {
			maxTokens = budget + defaultAnthropicMaxTokens
		}
		maxTokens = clampMaxTokensToContext(model, tc.Messages, maxTokens)
		o.MaxTokens = maxTokens
		o.ThinkingBudgetTokens = min(budget, max(0, maxTokens-minAnswerTokens))
	}
	return streamAnthropic(ctx, model, tc, o, level)
}

// anthropicEffort maps a level to an output_config effort: the model's mapping
// when it has one, otherwise the nearest of low, medium and high.
func anthropicEffort(model aimix.Model, level aimix.ThinkingLevel) string {
	if mapped, ok := model.ThinkingLevels.Lookup(level); ok && mapped != "" {
		return mapped
	}
	switch level {
	case aimix.ThinkingMinimal, aimix.ThinkingLow:
		return "low"
	case aimix.ThinkingMedium:
		return "medium"
	default:
		return "high"
	}
}

func streamAnthropic(ctx context.Context, model aimix.Model, tc aimix.TranscriptContext, opts AnthropicOptions, level aimix.ThinkingLevel) (*aimix.AssistantMessageEventStream, error) {
	if opts.APIKey == "" {
		return nil, errMissingAPIKey
	}
	params, betas, err := buildAnthropicParams(model, tc, opts)
	if err != nil {
		return nil, err
	}
	var extra []anthropicoption.RequestOption
	if len(betas) > 0 {
		extra = append(extra, anthropicoption.WithHeader("anthropic-beta", strings.Join(betas, ",")))
	}
	client := newAnthropicClient(model, opts.StreamOptions, extra...)

	w := newStreamWriter(ctx, model)
	w.out.ThinkingLevel = level
	return w.run(func() error {
		var resp *http.Response
		stream := client.Messages.NewStreaming(ctx, params, anthropicoption.WithResponseInto(&resp))
		defer stream.Close()
		if err := stream.Err(); err != nil {
			return err
		}
		if err := checkEventStream(resp); err != nil {
			return err
		}
		w.start()
		return processAnthropicStream(w, stream)
	}), nil
}

func buildAnthropicParams(model aimix.Model, tc aimix.TranscriptContext, opts AnthropicOptions) (anthropic.MessageNewParams, []string, error) {
	system, rest := splitSystem(aimix.CollapseSystemMessages(tc).Messages)
	messages := transformMessages(rest, model, normalizeAnthropicToolCallID)
	cacheControl := anthropicCacheControl(opts.CacheRetention)

	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = model.MaxTokens
	}
	if maxTokens <= 0 {
		maxTokens = defaultAnthropicMaxTokens
	}
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model.ID),
		MaxTokens: int64(maxTokens),
		Messages:  convertAnthropicMessages(messages, cacheControl),
	}

	if text := aimix.TextOf(system.Content); text != "" {
		block := anthropic.TextBlockParam{
			Text: text,
		}
		if cacheControl != nil {
			block.CacheControl = *cacheControl
		}
		params.System = []anthropic.TextBlockParam{block}
	}

	if len(system.ToolsAdded) > 0 {
		tools, err := convertAnthropicTools(system.ToolsAdded, cacheControl)
		if err != nil {
			return params, nil, err
		}
		params.Tools = tools
	}

	// Anthropic expresses ParallelToolCalls as tool_choice.disable_parallel_tool_use,
	// and only accepts tool_choice together with tools.
	noParallel := len(params.Tools) > 0 && opts.ParallelToolCalls != nil && !*opts.ParallelToolCalls
	switch {
	case opts.ToolChoice == aimix.ToolChoiceNone:
		params.ToolChoice = anthropic.ToolChoiceUnionParam{
			OfNone: &anthropic.ToolChoiceNoneParam{},
		}
	case opts.ToolChoice == aimix.ToolChoiceRequired:
		choice := anthropic.ToolChoiceAnyParam{}
		if noParallel {
			choice.DisableParallelToolUse = anthropic.Bool(true)
		}
		params.ToolChoice = anthropic.ToolChoiceUnionParam{
			OfAny: &choice,
		}
	case opts.ToolChoice == aimix.ToolChoiceAuto || noParallel:
		choice := anthropic.ToolChoiceAutoParam{}
		if noParallel {
			choice.DisableParallelToolUse = anthropic.Bool(true)
		}
		params.ToolChoice = anthropic.ToolChoiceUnionParam{
			OfAuto: &choice,
		}
	}

	var betas []string
	thinkingOn := model.Reasoning && opts.ThinkingEnabled != nil && *opts.ThinkingEnabled
	off, offMapped := model.ThinkingLevels.Lookup(aimix.ThinkingOff)
	thinkingOff := model.Reasoning && opts.ThinkingEnabled != nil && !*opts.ThinkingEnabled && !(offMapped && off == "")
	switch {
	case thinkingOff:
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfDisabled: &anthropic.ThinkingConfigDisabledParam{},
		}
	case thinkingOn && model.Compat.AdaptiveThinking:
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{
				Display: anthropic.ThinkingConfigAdaptiveDisplaySummarized,
			},
		}
		if opts.Effort != "" {
			params.OutputConfig = anthropic.OutputConfigParam{
				Effort: anthropic.OutputConfigEffort(opts.Effort),
			}
		}
	case thinkingOn:
		budget := opts.ThinkingBudgetTokens
		if budget <= 0 {
			budget = minAnthropicThinkingBudget
		}
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{
				BudgetTokens: int64(budget),
				Display:      anthropic.ThinkingConfigEnabledDisplaySummarized,
			},
		}
		betas = append(betas, interleavedThinkingBeta)
	}

	// Temperature is incompatible with thinking.
	if opts.Temperature != nil && !thinkingOn && !model.Compat.NoTemperature {
		params.Temperature = anthropic.Float(*opts.Temperature)
	}
	return params, betas, nil
}

// normalizeAnthropicToolCallID makes a foreign tool call id match the
// ^[a-zA-Z0-9_-]{1,64}$ pattern the API requires.
func normalizeAnthropicToolCallID(id string) string {
	return sanitizeID(id, 64)
}

func anthropicCacheControl(retention aimix.CacheRetention) *anthropic.CacheControlEphemeralParam {
	switch retention {
	case aimix.CacheRetentionNone:
		return nil
	case aimix.CacheRetentionLong:
		cc := anthropic.CacheControlEphemeralParam{
			TTL: anthropic.CacheControlEphemeralTTLTTL1h,
		}
		return &cc
	}
	cc := anthropic.NewCacheControlEphemeralParam()
	return &cc
}

func convertAnthropicTools(tools []aimix.Tool, cacheControl *anthropic.CacheControlEphemeralParam) ([]anthropic.ToolUnionParam, error) {
	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for i, t := range tools {
		schema, err := toolSchema(t)
		if err != nil {
			return nil, err
		}
		tool := anthropic.ToolParam{
			Name:                t.Name,
			InputSchema:         anthropicparam.Override[anthropic.ToolInputSchemaParam](schema),
			EagerInputStreaming: anthropic.Bool(true),
		}
		if t.Description != "" {
			tool.Description = anthropic.String(t.Description)
		}
		if t.Strict {
			tool.Strict = anthropic.Bool(true)
		}
		// The cache breakpoint on the last tool covers the whole tool list.
		if cacheControl != nil && i == len(tools)-1 {
			tool.CacheControl = *cacheControl
		}
		out = append(out, anthropic.ToolUnionParam{
			OfTool: &tool,
		})
	}
	return out, nil
}

// toolSchema returns the tool's parameters schema, defaulting to an empty object
// schema.
func toolSchema(t aimix.Tool) (json.RawMessage, error) {
	if len(t.Parameters) == 0 {
		return json.RawMessage(`{"type":"object","properties":{}}`), nil
	}
	if !json.Valid(t.Parameters) {
		return nil, fmt.Errorf("aimix: tool %q: parameters are not valid JSON", t.Name)
	}
	return t.Parameters, nil
}

func convertAnthropicMessages(messages []aimix.Message, cacheControl *anthropic.CacheControlEphemeralParam) []anthropic.MessageParam {
	var params []anthropic.MessageParam
	for i := 0; i < len(messages); i++ {
		switch m := messages[i].(type) {
		case aimix.UserMessage:
			var blocks []anthropic.ContentBlockParamUnion
			for _, c := range m.Content {
				switch b := c.(type) {
				case aimix.TextContent:
					if strings.TrimSpace(b.Text) != "" {
						blocks = append(blocks, anthropic.NewTextBlock(b.Text))
					}
				case aimix.ImageContent:
					blocks = append(blocks, anthropic.NewImageBlockBase64(b.MimeType, b.Data))
				}
			}
			if len(blocks) > 0 {
				params = append(params, anthropic.NewUserMessage(blocks...))
			}

		case aimix.AssistantMessage:
			if blocks := anthropicAssistantBlocks(m); len(blocks) > 0 {
				params = append(params, anthropic.NewAssistantMessage(blocks...))
			}

		case aimix.ToolResultMessage:
			// Consecutive tool results go into one user message. Images of error
			// results follow all the results, because the API rejects images in
			// error results.
			var results, errorImages []anthropic.ContentBlockParamUnion
			j := i
			for ; j < len(messages); j++ {
				tr, ok := messages[j].(aimix.ToolResultMessage)
				if !ok {
					break
				}
				content := tr.Content
				if tr.IsError && hasImages(content) {
					errorImages = append(errorImages, anthropic.NewTextBlock(
						fmt.Sprintf("Images from the error result of tool call %s (%s):", tr.ToolCallID, tr.ToolName),
					))
					for _, c := range content {
						if img, ok := c.(aimix.ImageContent); ok {
							errorImages = append(errorImages, anthropic.NewImageBlockBase64(img.MimeType, img.Data))
						}
					}
					content = replaceImages(content, errorToolImagePlaceholder)
				}
				results = append(results, anthropicToolResult(tr, content))
			}
			i = j - 1
			params = append(params, anthropic.NewUserMessage(append(results, errorImages...)...))
		}
	}

	if cacheControl != nil && len(params) > 0 {
		last := params[len(params)-1]
		if last.Role == anthropic.MessageParamRoleUser && len(last.Content) > 0 {
			setAnthropicCacheControl(&last.Content[len(last.Content)-1], *cacheControl)
		}
	}
	return params
}

func anthropicAssistantBlocks(m aimix.AssistantMessage) []anthropic.ContentBlockParamUnion {
	var blocks []anthropic.ContentBlockParamUnion
	for _, c := range m.Content {
		switch b := c.(type) {
		case aimix.TextContent:
			if strings.TrimSpace(b.Text) != "" {
				blocks = append(blocks, anthropic.NewTextBlock(b.Text))
			}
		case aimix.ThinkingContent:
			if b.Redacted {
				blocks = append(blocks, anthropic.NewRedactedThinkingBlock(b.ThinkingSignature))
				continue
			}
			hasSignature := strings.TrimSpace(b.ThinkingSignature) != ""
			switch {
			case hasSignature:
				blocks = append(blocks, anthropic.NewThinkingBlock(b.ThinkingSignature, b.Thinking))
			case strings.TrimSpace(b.Thinking) != "":
				// Thinking without a signature cannot be replayed as thinking.
				blocks = append(blocks, anthropic.NewTextBlock(b.Thinking))
			}
		case aimix.ToolCallContent:
			args := b.Arguments
			if len(args) == 0 || !json.Valid(args) {
				args = json.RawMessage("{}")
			}
			blocks = append(blocks, anthropic.NewToolUseBlock(b.ID, args, b.Name))
		}
	}
	return blocks
}

func anthropicToolResult(tr aimix.ToolResultMessage, content aimix.Contents) anthropic.ContentBlockParamUnion {
	block := anthropic.ToolResultBlockParam{
		ToolUseID: tr.ToolCallID,
		IsError:   anthropic.Bool(tr.IsError),
	}
	if !hasImages(content) {
		if text := strings.Join(textBlocks(content), "\n"); text != "" {
			block.Content = []anthropic.ToolResultBlockParamContentUnion{
				{
					OfText: &anthropic.TextBlockParam{
						Text: text,
					},
				},
			}
		}
	} else {
		hasText := false
		for _, c := range content {
			switch b := c.(type) {
			case aimix.TextContent:
				hasText = true
				block.Content = append(block.Content, anthropic.ToolResultBlockParamContentUnion{
					OfText: &anthropic.TextBlockParam{
						Text: b.Text,
					},
				})
			case aimix.ImageContent:
				block.Content = append(block.Content, anthropic.ToolResultBlockParamContentUnion{
					OfImage: &anthropic.ImageBlockParam{
						Source: anthropic.ImageBlockParamSourceUnion{
							OfBase64: &anthropic.Base64ImageSourceParam{
								Data:      b.Data,
								MediaType: anthropic.Base64ImageSourceMediaType(b.MimeType),
							},
						},
					},
				})
			}
		}
		if !hasText {
			block.Content = append([]anthropic.ToolResultBlockParamContentUnion{
				{
					OfText: &anthropic.TextBlockParam{
						Text: "(see attached image)",
					},
				},
			}, block.Content...)
		}
	}
	return anthropic.ContentBlockParamUnion{
		OfToolResult: &block,
	}
}

func setAnthropicCacheControl(block *anthropic.ContentBlockParamUnion, cc anthropic.CacheControlEphemeralParam) {
	switch {
	case block.OfText != nil:
		block.OfText.CacheControl = cc
	case block.OfImage != nil:
		block.OfImage.CacheControl = cc
	case block.OfToolResult != nil:
		block.OfToolResult.CacheControl = cc
	}
}

func processAnthropicStream(w *streamWriter, stream *anthropicssestream.Stream[anthropic.MessageStreamEventUnion]) error {
	var (
		blocks    = make(map[int64]int)           // Anthropic block index -> content index
		toolInput = make(map[int]json.RawMessage) // input sent in content_block_start
		started   bool
		stopped   bool
		stopErr   error
	)
	for stream.Next() {
		switch e := stream.Current().AsAny().(type) {
		case anthropic.MessageStartEvent:
			started = true
			w.out.ResponseID = e.Message.ID
			if m := e.Message.Model; m != "" && m != w.out.Model {
				w.out.ResponseModel = m
			}
			applyAnthropicStartUsage(&w.out.Usage, e.Message.Usage)

		case anthropic.ContentBlockStartEvent:
			switch b := e.ContentBlock.AsAny().(type) {
			case anthropic.TextBlock:
				i := w.startText()
				blocks[e.Index] = i
				w.appendText(i, b.Text)
			case anthropic.ThinkingBlock:
				i := w.startThinking()
				blocks[e.Index] = i
				w.appendThinking(i, b.Thinking)
				w.appendThinkingSignature(i, b.Signature)
			case anthropic.RedactedThinkingBlock:
				i := w.startThinking()
				blocks[e.Index] = i
				w.setThinking(i, func(c *aimix.ThinkingContent) {
					c.Thinking = "[Reasoning redacted]"
					c.ThinkingSignature = b.Data
					c.Redacted = true
				})
			case anthropic.ToolUseBlock:
				i := w.startToolCall(b.ID, b.Name)
				blocks[e.Index] = i
				toolInput[i] = b.Input
			}

		case anthropic.ContentBlockDeltaEvent:
			i, ok := blocks[e.Index]
			if !ok {
				continue
			}
			switch d := e.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				w.appendText(i, d.Text)
			case anthropic.ThinkingDelta:
				w.appendThinking(i, d.Thinking)
			case anthropic.SignatureDelta:
				w.appendThinkingSignature(i, d.Signature)
			case anthropic.InputJSONDelta:
				w.appendToolArgs(i, d.PartialJSON)
			}

		case anthropic.ContentBlockStopEvent:
			i, ok := blocks[e.Index]
			if !ok {
				continue
			}
			delete(blocks, e.Index)
			switch w.out.Content[i].(type) {
			case aimix.TextContent:
				w.endText(i)
			case aimix.ThinkingContent:
				w.endThinking(i)
			case aimix.ToolCallContent:
				// Without streamed input, the input of content_block_start is final.
				if w.args[i].Text() == "" {
					if in := toolInput[i]; len(in) > 0 && string(in) != "{}" {
						w.replaceToolArgs(i, string(in))
					}
				}
				w.endToolCall(i)
			}

		case anthropic.MessageDeltaEvent:
			if e.Delta.StopReason != "" {
				w.out.RawStopReason = string(e.Delta.StopReason)
				w.out.StopReason, stopErr = mapAnthropicStopReason(e.Delta)
			}
			applyAnthropicDeltaUsage(&w.out.Usage, e.Usage)

		case anthropic.MessageStopEvent:
			stopped = true
		}
	}
	if err := stream.Err(); err != nil {
		return err
	}
	if started && !stopped {
		return errors.New("aimix: Anthropic stream ended before message_stop")
	}
	return stopErr
}

func mapAnthropicStopReason(d anthropic.MessageDeltaEventDelta) (aimix.StopReason, error) {
	switch d.StopReason {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonPauseTurn, anthropic.StopReasonStopSequence:
		return aimix.StopReasonStop, nil
	case anthropic.StopReasonMaxTokens, "model_context_window_exceeded":
		return aimix.StopReasonLength, nil
	case anthropic.StopReasonToolUse:
		return aimix.StopReasonToolUse, nil
	case anthropic.StopReasonRefusal:
		if d.StopDetails.Explanation != "" {
			return aimix.StopReasonError, errors.New(d.StopDetails.Explanation)
		}
		return aimix.StopReasonError, errors.New("the model refused to respond")
	}
	return aimix.StopReasonError, fmt.Errorf("aimix: unhandled Anthropic stop reason %q", d.StopReason)
}

func applyAnthropicStartUsage(u *aimix.Usage, s anthropic.Usage) {
	u.Input = int(s.InputTokens)
	u.Output = int(s.OutputTokens)
	u.CacheRead = int(s.CacheReadInputTokens)
	u.CacheWrite = int(s.CacheCreationInputTokens)
	u.CacheWrite1h = int(s.CacheCreation.Ephemeral1hInputTokens)
	if s.JSON.OutputTokensDetails.Valid() {
		u.Reasoning = int(s.OutputTokensDetails.ThinkingTokens)
	}
	u.TotalTokens = u.Input + u.Output + u.CacheRead + u.CacheWrite
}

// applyAnthropicDeltaUsage applies the cumulative counts of message_delta. Only
// fields the event carries are overridden.
func applyAnthropicDeltaUsage(u *aimix.Usage, d anthropic.MessageDeltaUsage) {
	if d.JSON.InputTokens.Valid() {
		u.Input = int(d.InputTokens)
	}
	if d.JSON.OutputTokens.Valid() {
		u.Output = int(d.OutputTokens)
	}
	if d.JSON.CacheReadInputTokens.Valid() {
		u.CacheRead = int(d.CacheReadInputTokens)
	}
	if d.JSON.CacheCreationInputTokens.Valid() {
		u.CacheWrite = int(d.CacheCreationInputTokens)
	}
	if d.JSON.OutputTokensDetails.Valid() {
		u.Reasoning = int(d.OutputTokensDetails.ThinkingTokens)
	}
	u.TotalTokens = u.Input + u.Output + u.CacheRead + u.CacheWrite
}

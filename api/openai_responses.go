package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"net/http"
	"strconv"
	"strings"

	"github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
	openaiparam "github.com/openai/openai-go/v3/packages/param"
	openaissestream "github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/yoneyy/aimix"
)

const (
	// minResponsesOutputTokens is the smallest max_output_tokens the API accepts.
	minResponsesOutputTokens = 16
	maxPromptCacheKeyLen     = 64
	reasoningEffortNone      = "none"
)

// OpenAIResponsesOptions are the options of the openai-responses Api.
type OpenAIResponsesOptions struct {
	aimix.StreamOptions
	// ReasoningEffort is the reasoning.effort wire value ("minimal", "low",
	// "medium", "high", "xhigh", "none", ...). Empty leaves the model default.
	ReasoningEffort string
	// ReasoningSummary is the reasoning.summary mode. Empty means "auto".
	ReasoningSummary string
}

type openAIResponses struct{}

// OpenAIResponses returns the openai-responses implementation.
func OpenAIResponses() aimix.ApiStreamer {
	return openAIResponses{}
}

func (openAIResponses) Stream(ctx context.Context, req aimix.StreamRequest) (*aimix.AssistantMessageEventStream, error) {
	return StreamOpenAIResponses(ctx, req.Model, req.Context, OpenAIResponsesOptions{
		StreamOptions: req.Options,
	})
}

func (openAIResponses) StreamSimple(ctx context.Context, req aimix.SimpleStreamRequest) (*aimix.AssistantMessageEventStream, error) {
	return StreamSimpleOpenAIResponses(ctx, req.Model, req.Context, req.Options)
}

// StreamOpenAIResponses streams one assistant turn from the OpenAI Responses API.
// Requests are stateless (store=false): reasoning is replayed from the encrypted
// content kept in ThinkingContent.ThinkingSignature.
func StreamOpenAIResponses(ctx context.Context, model aimix.Model, tc aimix.TranscriptContext, opts OpenAIResponsesOptions) (*aimix.AssistantMessageEventStream, error) {
	return streamResponses(ctx, model, tc, opts, "")
}

// StreamSimpleOpenAIResponses maps the unified thinking level to reasoning.effort.
// Off sends the model's mapped "off" value, or nothing when it has none.
func StreamSimpleOpenAIResponses(ctx context.Context, model aimix.Model, tc aimix.TranscriptContext, opts aimix.SimpleStreamOptions) (*aimix.AssistantMessageEventStream, error) {
	level := effectiveReasoning(model, opts.Reasoning)
	o := OpenAIResponsesOptions{
		StreamOptions: opts.StreamOptions,
	}
	o.MaxTokens = baseMaxTokens(model, tc.Messages, opts.StreamOptions)
	o.ReasoningEffort = openAIEffort(model, level)
	return streamResponses(ctx, model, tc, o, level)
}

// openAIEffort maps a level to a reasoning effort: the model's mapping when it
// has one, otherwise the level name. Off maps to nothing unless mapped.
func openAIEffort(model aimix.Model, level aimix.ThinkingLevel) string {
	mapped, ok := model.ThinkingLevels.Lookup(level)
	switch {
	case ok:
		return mapped
	case level == aimix.ThinkingOff || !model.Reasoning:
		return ""
	}
	return string(level)
}

func streamResponses(ctx context.Context, model aimix.Model, tc aimix.TranscriptContext, opts OpenAIResponsesOptions, level aimix.ThinkingLevel) (*aimix.AssistantMessageEventStream, error) {
	if opts.APIKey == "" {
		return nil, errMissingAPIKey
	}
	params, err := buildResponsesParams(model, tc, opts)
	if err != nil {
		return nil, err
	}
	service := responses.NewResponseService(openAIRequestOptions(model, opts.StreamOptions)...)

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
		return processResponsesStream(w, stream)
	}), nil
}

func buildResponsesParams(model aimix.Model, tc aimix.TranscriptContext, opts OpenAIResponsesOptions) (responses.ResponseNewParams, error) {
	system, rest := splitSystem(aimix.CollapseSystemMessages(tc).Messages)
	messages := transformMessages(rest, model, normalizeResponsesToolCallID)

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(model.ID),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: convertResponsesInput(model, system, messages),
		},
		Store: openai.Bool(false),
	}

	if opts.CacheRetention != aimix.CacheRetentionNone && opts.SessionID != "" {
		key := opts.SessionID
		if len(key) > maxPromptCacheKeyLen {
			key = key[:maxPromptCacheKeyLen]
		}
		params.PromptCacheKey = openai.String(key)
	}
	if opts.CacheRetention == aimix.CacheRetentionLong {
		params.PromptCacheRetention = responses.ResponseNewParamsPromptCacheRetention24h
	}
	if opts.MaxTokens > 0 {
		params.MaxOutputTokens = openai.Int(int64(max(opts.MaxTokens, minResponsesOutputTokens)))
	}
	if opts.Temperature != nil && !model.Compat.NoTemperature {
		params.Temperature = openai.Float(*opts.Temperature)
	}

	if len(system.ToolsAdded) > 0 {
		tools := make([]responses.ToolUnionParam, 0, len(system.ToolsAdded))
		for _, t := range system.ToolsAdded {
			schema, err := toolSchemaMap(t)
			if err != nil {
				return params, err
			}
			// The API defaults strict to true, so it is always sent.
			fn := responses.FunctionToolParam{
				Name:       t.Name,
				Parameters: schema,
				Strict:     openai.Bool(t.Strict),
			}
			if t.Description != "" {
				fn.Description = openai.String(t.Description)
			}
			tools = append(tools, responses.ToolUnionParam{
				OfFunction: &fn,
			})
		}
		params.Tools = tools
	}
	if mode, ok := responsesToolChoice(opts.ToolChoice); ok {
		params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: openaiparam.NewOpt(mode),
		}
	}
	// parallel_tool_calls is only accepted together with tools.
	if opts.ParallelToolCalls != nil && len(params.Tools) > 0 {
		params.ParallelToolCalls = openai.Bool(*opts.ParallelToolCalls)
	}

	effort := opts.ReasoningEffort
	if effort == "" && opts.ReasoningSummary != "" {
		effort = "medium"
	}
	if model.Reasoning && effort != "" {
		params.Reasoning = shared.ReasoningParam{
			Effort: shared.ReasoningEffort(effort),
		}
		// Reasoning that is turned off has nothing to summarize or replay.
		if effort != reasoningEffortNone {
			summary := opts.ReasoningSummary
			if summary == "" {
				summary = "auto"
			}
			params.Reasoning.Summary = shared.ReasoningSummary(summary)
			params.Include = []responses.ResponseIncludable{
				responses.ResponseIncludableReasoningEncryptedContent,
			}
		}
	}
	return params, nil
}

func responsesToolChoice(choice aimix.ToolChoice) (responses.ToolChoiceOptions, bool) {
	switch choice {
	case aimix.ToolChoiceAuto:
		return responses.ToolChoiceOptionsAuto, true
	case aimix.ToolChoiceNone:
		return responses.ToolChoiceOptionsNone, true
	case aimix.ToolChoiceRequired:
		return responses.ToolChoiceOptionsRequired, true
	}
	return "", false
}

// toolSchemaMap decodes the tool's parameters schema into the map form the
// OpenAI SDK takes.
func toolSchemaMap(t aimix.Tool) (map[string]any, error) {
	raw, err := toolSchema(t)
	if err != nil {
		return nil, err
	}
	var schema map[string]any
	if err := json.Unmarshal(raw, &schema); err != nil {
		return nil, fmt.Errorf("aimix: tool %q: parameters must be a JSON object: %w", t.Name, err)
	}
	return schema, nil
}

// normalizeResponsesToolCallID rewrites a foreign tool call id. Ids of the form
// "call_id|item_id" keep both parts; the item id gets a stable "fc_" id, since
// the API only accepts function call item ids it could have issued.
func normalizeResponsesToolCallID(id string) string {
	callID, itemID, ok := strings.Cut(id, "|")
	if !ok {
		return normalizeIDPart(id)
	}
	return normalizeIDPart(callID) + "|" + foreignItemID(itemID)
}

func normalizeIDPart(part string) string {
	return strings.TrimRight(sanitizeID(part, 64), "_")
}

func foreignItemID(itemID string) string {
	id := "fc_" + shortHash(itemID)
	if len(id) > 64 {
		id = id[:64]
	}
	return id
}

func shortHash(s string) string {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return strconv.FormatUint(h.Sum64(), 36)
}

// textSignatureV1 is the TextSignature of Responses output messages: the
// message item id and phase, needed to replay the message.
type textSignatureV1 struct {
	V     int    `json:"v"`
	ID    string `json:"id"`
	Phase string `json:"phase,omitempty"`
}

func encodeTextSignature(id, phase string) string {
	raw, _ := json.Marshal(textSignatureV1{
		V:     1,
		ID:    id,
		Phase: phase,
	})
	return string(raw)
}

// parseTextSignature accepts the V1 JSON form and a bare message id.
func parseTextSignature(sig string) (textSignatureV1, bool) {
	if sig == "" {
		return textSignatureV1{}, false
	}
	if strings.HasPrefix(sig, "{") {
		var v textSignatureV1
		if json.Unmarshal([]byte(sig), &v) == nil && v.V == 1 && v.ID != "" {
			if v.Phase != "commentary" && v.Phase != "final_answer" {
				v.Phase = ""
			}
			return v, true
		}
	}
	return textSignatureV1{
		V:  1,
		ID: sig,
	}, true
}

func convertResponsesInput(model aimix.Model, system aimix.SystemMessage, messages []aimix.Message) responses.ResponseInputParam {
	var input responses.ResponseInputParam
	if text := aimix.TextOf(system.Content); text != "" {
		role := responses.EasyInputMessageRoleSystem
		if model.Reasoning {
			role = responses.EasyInputMessageRoleDeveloper
		}
		input = append(input, responses.ResponseInputItemParamOfMessage(text, role))
	}

	for msgIndex, m := range messages {
		switch v := m.(type) {
		case aimix.UserMessage:
			var parts responses.ResponseInputMessageContentListParam
			for _, c := range v.Content {
				switch b := c.(type) {
				case aimix.TextContent:
					parts = append(parts, responses.ResponseInputContentUnionParam{
						OfInputText: &responses.ResponseInputTextParam{
							Text: b.Text,
						},
					})
				case aimix.ImageContent:
					parts = append(parts, responses.ResponseInputContentUnionParam{
						OfInputImage: &responses.ResponseInputImageParam{
							ImageURL: openai.String(dataURL(b)),
							Detail:   responses.ResponseInputImageDetailAuto,
						},
					})
				}
			}
			if len(parts) > 0 {
				input = append(input, responses.ResponseInputItemParamOfMessage(parts, responses.EasyInputMessageRoleUser))
			}

		case aimix.AssistantMessage:
			input = append(input, responsesAssistantItems(model, v, msgIndex)...)

		case aimix.ToolResultMessage:
			callID, _, _ := strings.Cut(v.ToolCallID, "|")
			input = append(input, responses.ResponseInputItemUnionParam{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					CallID: openai.String(callID),
					Output: responsesToolOutput(model, v.Content),
				},
			})
		}
	}
	return input
}

func responsesAssistantItems(model aimix.Model, m aimix.AssistantMessage, msgIndex int) []responses.ResponseInputItemUnionParam {
	sameProviderAndApi := m.Provider == model.Provider && m.Api == model.Api
	sameModel := sameProviderAndApi && m.Model == model.ID
	differentModel := sameProviderAndApi && m.Model != model.ID

	var items []responses.ResponseInputItemUnionParam
	textIndex := 0
	for _, c := range m.Content {
		switch b := c.(type) {
		case aimix.ThinkingContent:
			// The signature is the reasoning item as received, encrypted content included.
			if b.ThinkingSignature != "" && json.Valid([]byte(b.ThinkingSignature)) {
				reasoning := openaiparam.Override[responses.ResponseReasoningItemParam](json.RawMessage(b.ThinkingSignature))
				items = append(items, responses.ResponseInputItemUnionParam{
					OfReasoning: &reasoning,
				})
			}

		case aimix.TextContent:
			id := fmt.Sprintf("msg_aimix_%d", msgIndex)
			if textIndex > 0 {
				id = fmt.Sprintf("msg_aimix_%d_%d", msgIndex, textIndex)
			}
			textIndex++
			var phase string
			if sig, ok := parseTextSignature(b.TextSignature); ok {
				id, phase = sig.ID, sig.Phase
				if len(id) > 64 {
					id = "msg_" + shortHash(id)
				}
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfOutputMessage: &responses.ResponseOutputMessageParam{
					ID: id,
					Content: []responses.ResponseOutputMessageContentUnionParam{
						{
							OfOutputText: &responses.ResponseOutputTextParam{
								Text:        b.Text,
								Annotations: []responses.ResponseOutputTextAnnotationUnionParam{},
							},
						},
					},
					Status: responses.ResponseOutputMessageStatusCompleted,
					Phase:  responses.ResponseOutputMessagePhase(phase),
				},
			})

		case aimix.ToolCallContent:
			callID, itemID, _ := strings.Cut(b.ID, "|")
			if differentModel || !strings.HasPrefix(itemID, "fc_") {
				itemID = ""
			}
			args := string(b.Arguments)
			if args == "" {
				args = "{}"
			}
			call := responses.ResponseFunctionToolCallParam{
				CallID:    callID,
				Name:      b.Name,
				Arguments: args,
			}
			if itemID != "" {
				call.ID = openai.String(itemID)
			}
			if sameModel && b.Namespace != "" {
				call.Namespace = openai.String(b.Namespace)
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfFunctionCall: &call,
			})
		}
	}
	return items
}

func responsesToolOutput(model aimix.Model, content aimix.Contents) responses.ResponseInputItemFunctionCallOutputOutputUnionParam {
	text := strings.Join(textBlocks(content), "\n")
	hasText := strings.TrimSpace(text) != ""
	images := hasImages(content)

	if !images || !model.SupportsImages() {
		switch {
		case hasText:
		case images:
			text = "(see attached image)"
		default:
			text = "(no tool output)"
		}
		return responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfString: openai.String(text),
		}
	}

	var items responses.ResponseFunctionCallOutputItemListParam
	if hasText {
		items = append(items, responses.ResponseFunctionCallOutputItemUnionParam{
			OfInputText: &responses.ResponseInputTextContentParam{
				Text: text,
			},
		})
	}
	for _, c := range content {
		if img, ok := c.(aimix.ImageContent); ok {
			items = append(items, responses.ResponseFunctionCallOutputItemUnionParam{
				OfInputImage: &responses.ResponseInputImageContentParam{
					ImageURL: openai.String(dataURL(img)),
					Detail:   responses.ResponseInputImageContentDetailAuto,
				},
			})
		}
	}
	return responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
		OfResponseFunctionCallOutputItemArray: items,
	}
}

func dataURL(img aimix.ImageContent) string {
	return "data:" + img.MimeType + ";base64," + img.Data
}

func processResponsesStream(w *streamWriter, stream *openaissestream.Stream[responses.ResponseStreamEventUnion]) error {
	var (
		slots      = make(map[int64]int)    // output_index -> content index
		reasoning  = make(map[string]int)   // reasoning item id -> content index
		unfinished = make(map[int]struct{}) // tool calls without output_item.done
		terminal   bool
		stopErr    error
	)
	slotOf := func(outputIndex int64, kind aimix.ContentType) (int, bool) {
		i, ok := slots[outputIndex]
		if !ok || contentType(w.out.Content[i]) != kind {
			return 0, false
		}
		return i, true
	}
	createSlot := func(outputIndex int64, item responses.ResponseOutputItemUnion) (int, bool) {
		var i int
		switch item.Type {
		case "reasoning":
			i = w.startThinking()
		case "message":
			i = w.startText()
		case "function_call":
			i = w.startToolCall(item.CallID+"|"+item.ID, item.Name)
			w.appendToolArgs(i, item.Arguments.OfString)
			unfinished[i] = struct{}{}
		default:
			return 0, false
		}
		slots[outputIndex] = i
		return i, true
	}

	for stream.Next() {
		switch e := stream.Current().AsAny().(type) {
		case responses.ResponseCreatedEvent:
			w.out.ResponseID = e.Response.ID

		case responses.ResponseOutputItemAddedEvent:
			createSlot(e.OutputIndex, e.Item)

		case responses.ResponseReasoningSummaryTextDeltaEvent:
			if i, ok := slotOf(e.OutputIndex, aimix.ContentTypeThinking); ok {
				w.appendThinking(i, e.Delta)
			}
		case responses.ResponseReasoningSummaryPartDoneEvent:
			if i, ok := slotOf(e.OutputIndex, aimix.ContentTypeThinking); ok {
				w.appendThinking(i, "\n\n")
			}
		case responses.ResponseReasoningTextDeltaEvent:
			if i, ok := slotOf(e.OutputIndex, aimix.ContentTypeThinking); ok {
				w.appendThinking(i, e.Delta)
			}

		case responses.ResponseTextDeltaEvent:
			if i, ok := slotOf(e.OutputIndex, aimix.ContentTypeText); ok {
				w.appendText(i, e.Delta)
			}
		case responses.ResponseRefusalDeltaEvent:
			if i, ok := slotOf(e.OutputIndex, aimix.ContentTypeText); ok {
				w.appendText(i, e.Delta)
			}

		case responses.ResponseFunctionCallArgumentsDeltaEvent:
			if i, ok := slotOf(e.OutputIndex, aimix.ContentTypeToolCall); ok {
				w.appendToolArgs(i, e.Delta)
			}
		case responses.ResponseFunctionCallArgumentsDoneEvent:
			if i, ok := slotOf(e.OutputIndex, aimix.ContentTypeToolCall); ok {
				w.replaceToolArgs(i, e.Arguments)
			}

		case responses.ResponseOutputItemDoneEvent:
			item := e.Item
			i, ok := slots[e.OutputIndex]
			if !ok {
				if i, ok = createSlot(e.OutputIndex, item); !ok {
					continue
				}
			}
			delete(slots, e.OutputIndex)
			switch item.Type {
			case "reasoning":
				thinking := joinReasoning(item)
				w.setThinking(i, func(c *aimix.ThinkingContent) {
					if thinking != "" {
						c.Thinking = thinking
					}
					c.ThinkingSignature = item.RawJSON()
				})
				reasoning[item.ID] = i
				w.endThinking(i)
			case "message":
				var text strings.Builder
				for _, part := range item.Content {
					if part.Type == "output_text" {
						text.WriteString(part.Text)
					} else {
						text.WriteString(part.Refusal)
					}
				}
				update(w, i, func(c *aimix.TextContent) {
					c.Text = text.String()
					c.TextSignature = encodeTextSignature(item.ID, string(item.Phase))
				})
				w.endText(i)
			case "function_call":
				if item.Arguments.OfString != "" {
					w.replaceToolArgs(i, item.Arguments.OfString)
				}
				if item.Namespace != "" {
					w.updateToolCall(i, func(c *aimix.ToolCallContent) {
						c.Namespace = item.Namespace
					})
				}
				delete(unfinished, i)
				w.endToolCall(i)
			}

		case responses.ResponseCompletedEvent:
			terminal = true
			stopErr = finishResponse(w, e.Response, reasoning)
		case responses.ResponseIncompleteEvent:
			terminal = true
			stopErr = finishResponse(w, e.Response, reasoning)

		case responses.ResponseFailedEvent:
			w.out.RawStopReason = string(e.Response.Status)
			switch {
			case e.Response.Error.Message != "" || e.Response.Error.Code != "":
				return fmt.Errorf("%s: %s", e.Response.Error.Code, e.Response.Error.Message)
			case e.Response.IncompleteDetails.Reason != "":
				return fmt.Errorf("incomplete: %s", e.Response.IncompleteDetails.Reason)
			}
			return errors.New("response failed without error details")

		case responses.ResponseErrorEvent:
			return fmt.Errorf("OpenAI API error (%s): %s", e.Code, e.Message)
		}
	}
	if err := stream.Err(); err != nil {
		return err
	}
	if !terminal {
		return errors.New("aimix: OpenAI Responses stream ended before a terminal response event")
	}
	if stopErr != nil {
		return stopErr
	}
	for i := range unfinished {
		tc := w.out.Content[i].(aimix.ToolCallContent)
		return fmt.Errorf("aimix: OpenAI Responses stream completed with an unfinished tool call: %s (%s)", tc.Name, tc.ID)
	}
	return nil
}

func joinReasoning(item responses.ResponseOutputItemUnion) string {
	parts := make([]string, 0, len(item.Summary))
	for _, s := range item.Summary {
		parts = append(parts, s.Text)
	}
	if summary := strings.Join(parts, "\n\n"); summary != "" {
		return summary
	}
	parts = parts[:0]
	for _, c := range item.Content {
		parts = append(parts, c.Text)
	}
	return strings.Join(parts, "\n\n")
}

// finishResponse records usage and the stop reason of a completed or incomplete
// response, and adds encrypted reasoning that only the final output carries.
func finishResponse(w *streamWriter, resp responses.Response, reasoning map[string]int) error {
	for _, item := range resp.Output {
		if item.Type != "reasoning" || item.EncryptedContent == "" {
			continue
		}
		i, ok := reasoning[item.ID]
		if !ok {
			continue
		}
		w.setThinking(i, func(c *aimix.ThinkingContent) {
			c.ThinkingSignature = withEncryptedContent(c.ThinkingSignature, item.EncryptedContent)
		})
	}

	if resp.ID != "" {
		w.out.ResponseID = resp.ID
	}
	if m := string(resp.Model); m != "" && m != w.out.Model {
		w.out.ResponseModel = m
	}
	if resp.JSON.Usage.Valid() {
		u := resp.Usage
		cached := int(u.InputTokensDetails.CachedTokens)
		written := int(u.InputTokensDetails.CacheWriteTokens)
		w.out.Usage = aimix.Usage{
			Input:       max(0, int(u.InputTokens)-cached-written),
			Output:      int(u.OutputTokens),
			CacheRead:   cached,
			CacheWrite:  written,
			Reasoning:   int(u.OutputTokensDetails.ReasoningTokens),
			TotalTokens: int(u.TotalTokens),
		}
	}

	status := string(resp.Status)
	reason := resp.IncompleteDetails.Reason
	w.out.RawStopReason = status
	if reason != "" {
		w.out.RawStopReason = status + "." + reason
	}
	switch resp.Status {
	case responses.ResponseStatusIncomplete:
		if reason == "max_output_tokens" {
			w.out.StopReason = aimix.StopReasonLength
			return nil
		}
		w.out.StopReason = aimix.StopReasonError
		if reason == "" {
			return errors.New("response incomplete without a provider reason")
		}
		return fmt.Errorf("response incomplete: %s", reason)
	case responses.ResponseStatusFailed, responses.ResponseStatusCancelled:
		w.out.StopReason = aimix.StopReasonError
		return fmt.Errorf("response %s", status)
	}
	w.out.StopReason = aimix.StopReasonStop
	return nil
}

// withEncryptedContent sets encrypted_content in a serialized reasoning item that
// lacks it.
func withEncryptedContent(sig, encrypted string) string {
	var item map[string]any
	if json.Unmarshal([]byte(sig), &item) != nil {
		return sig
	}
	if s, _ := item["encrypted_content"].(string); s != "" {
		return sig
	}
	item["encrypted_content"] = encrypted
	raw, err := json.Marshal(item)
	if err != nil {
		return sig
	}
	return string(raw)
}

func contentType(c aimix.Content) aimix.ContentType {
	switch c.(type) {
	case aimix.TextContent:
		return aimix.ContentTypeText
	case aimix.ThinkingContent:
		return aimix.ContentTypeThinking
	case aimix.ImageContent:
		return aimix.ContentTypeImage
	case aimix.ToolCallContent:
		return aimix.ContentTypeToolCall
	}
	return ""
}

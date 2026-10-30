package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/yoneyy/aimix"
)

func responsesModel(baseURL string) aimix.Model {
	return aimix.Model{
		ID:            "gpt-test",
		Api:           aimix.ApiOpenAIResponses,
		Provider:      aimix.ProviderOpenAI,
		BaseURL:       baseURL,
		Reasoning:     true,
		ContextWindow: 400000,
		MaxTokens:     128000,
	}
}

const responsesReasoningItem = `{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Plan"}],"encrypted_content":"enc1"}`

var responsesToolTurn = strings.Join([]string{
	sseEvent("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":1,"status":"in_progress","model":"gpt-test","output":[]}}`),
	sseEvent("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`),
	sseEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","sequence_number":2,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"Plan"}`),
	sseEvent("response.reasoning_summary_part.done", `{"type":"response.reasoning_summary_part.done","sequence_number":3,"item_id":"rs_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":"Plan"}}`),
	sseEvent("response.output_item.done", `{"type":"response.output_item.done","sequence_number":4,"output_index":0,"item":`+responsesReasoningItem+`}`),
	sseEvent("response.output_item.added", `{"type":"response.output_item.added","sequence_number":5,"output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}`),
	sseEvent("response.output_text.delta", `{"type":"response.output_text.delta","sequence_number":6,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hi","logprobs":[]}`),
	sseEvent("response.output_item.done", `{"type":"response.output_item.done","sequence_number":7,"output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"commentary","content":[{"type":"output_text","text":"Hi","annotations":[]}]}}`),
	sseEvent("response.output_item.added", `{"type":"response.output_item.added","sequence_number":8,"output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"","status":"in_progress"}}`),
	sseEvent("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","sequence_number":9,"item_id":"fc_1","output_index":2,"delta":"{\"path\":"}`),
	sseEvent("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","sequence_number":10,"item_id":"fc_1","output_index":2,"delta":"\"a.txt\"}"}`),
	sseEvent("response.function_call_arguments.done", `{"type":"response.function_call_arguments.done","sequence_number":11,"item_id":"fc_1","output_index":2,"arguments":"{\"path\":\"a.txt\"}"}`),
	sseEvent("response.output_item.done", `{"type":"response.output_item.done","sequence_number":12,"output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"{\"path\":\"a.txt\"}","status":"completed"}}`),
	sseEvent("response.completed", `{"type":"response.completed","sequence_number":13,"response":{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-test-2025","output":[],`+
		`"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":40},"output_tokens":20,"output_tokens_details":{"reasoning_tokens":12},"total_tokens":120}}}`),
}, "")

func TestResponsesStreamToolTurn(t *testing.T) {
	srv, rec := serve(t, http.StatusOK, responsesToolTurn)
	model := responsesModel(srv.URL)
	tc := aimix.NormalizeContext(aimix.Context{
		SystemPrompt: "Be brief.",
		Messages: aimix.Messages{
			aimix.UserMessage{
				Role: aimix.RoleUser,
				Content: aimix.Contents{
					text("read a.txt"),
				},
			},
		},
		Tools: []aimix.Tool{
			{
				Name:        "read",
				Description: "Read a file",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
			},
		},
	})
	s, err := StreamSimpleOpenAIResponses(context.Background(), model, tc, aimix.SimpleStreamOptions{
		StreamOptions: aimix.StreamOptions{
			APIKey:    "sk-test",
			SessionID: "session-1",
		},
		Reasoning: aimix.ThinkingMedium,
	})
	if err != nil {
		t.Fatal(err)
	}
	events, msg := drain(t, s)

	wantEvents := "start,thinking_start,thinking_delta,thinking_delta,thinking_end,text_start,text_delta,text_end," +
		"tool_call_start,tool_call_delta,tool_call_delta,tool_call_end,done"
	if got := eventTypes(events); got != wantEvents {
		t.Fatalf("events:\n got %s\nwant %s", got, wantEvents)
	}
	if msg.StopReason != aimix.StopReasonToolUse || msg.RawStopReason != "completed" {
		t.Fatalf("stop reason %q (raw %q)", msg.StopReason, msg.RawStopReason)
	}
	if msg.ResponseID != "resp_1" || msg.ResponseModel != "gpt-test-2025" {
		t.Fatalf("metadata: %+v", msg)
	}
	wantUsage := aimix.Usage{
		Input:       60,
		Output:      20,
		CacheRead:   40,
		Reasoning:   12,
		TotalTokens: 120,
	}
	if msg.Usage != wantUsage {
		t.Fatalf("usage %+v, want %+v", msg.Usage, wantUsage)
	}

	th := msg.Content[0].(aimix.ThinkingContent)
	if th.Thinking != "Plan" || th.ThinkingSignature != responsesReasoningItem {
		t.Fatalf("thinking: %+v", th)
	}
	tx := msg.Content[1].(aimix.TextContent)
	if tx.Text != "Hi" || tx.TextSignature != `{"v":1,"id":"msg_1","phase":"commentary"}` {
		t.Fatalf("text: %+v", tx)
	}
	call := msg.Content[2].(aimix.ToolCallContent)
	if call.ID != "call_1|fc_1" || string(call.Arguments) != `{"path":"a.txt"}` {
		t.Fatalf("tool call: %+v", call)
	}

	body, header, path := rec.get()
	if path != "/responses" || header.Get("Authorization") != "Bearer sk-test" {
		t.Fatalf("path %q, headers %v", path, header)
	}
	checks := map[string]string{
		"store":             `false`,
		"prompt_cache_key":  `"session-1"`,
		"max_output_tokens": `128000`,
		"reasoning":         `{"effort":"medium","summary":"auto"}`,
		"include":           `["reasoning.encrypted_content"]`,
		"tools":             `[{"description":"Read a file","name":"read","parameters":{"properties":{"path":{"type":"string"}},"type":"object"},"strict":false,"type":"function"}]`,
		"input":             `[{"content":"Be brief.","role":"developer"},{"content":[{"text":"read a.txt","type":"input_text"}],"role":"user"}]`,
	}
	for field, want := range checks {
		if got := compactJSON(t, body[field]); got != want {
			t.Errorf("%s:\n got %s\nwant %s", field, got, want)
		}
	}
}

func TestResponsesConvertTranscript(t *testing.T) {
	model := responsesModel("")
	model.Input = []aimix.Modality{
		aimix.ModalityText,
		aimix.ModalityImage,
	}
	messages := aimix.Messages{
		aimix.AssistantMessage{
			Role:       aimix.RoleAssistant,
			Api:        aimix.ApiOpenAIResponses,
			Provider:   aimix.ProviderOpenAI,
			Model:      "gpt-test",
			StopReason: aimix.StopReasonToolUse,
			Content: aimix.Contents{
				aimix.ThinkingContent{
					Type:              aimix.ContentTypeThinking,
					Thinking:          "Plan",
					ThinkingSignature: responsesReasoningItem,
				},
				aimix.TextContent{
					Type:          aimix.ContentTypeText,
					Text:          "Hi",
					TextSignature: `{"v":1,"id":"msg_1","phase":"commentary"}`,
				},
				aimix.ToolCallContent{
					Type:      aimix.ContentTypeToolCall,
					ID:        "call_1|fc_1",
					Name:      "read",
					Arguments: json.RawMessage(`{"path":"a.txt"}`),
				},
			},
		},
		aimix.ToolResultMessage{
			Role:       aimix.RoleToolResult,
			ToolCallID: "call_1|fc_1",
			ToolName:   "read",
			Content: aimix.Contents{
				text("file body"),
				aimix.ImageContent{
					Type:     aimix.ContentTypeImage,
					Data:     "AAAA",
					MimeType: "image/png",
				},
			},
		},
		// Another model of the same provider and api: the item id is dropped.
		aimix.AssistantMessage{
			Role:       aimix.RoleAssistant,
			Api:        aimix.ApiOpenAIResponses,
			Provider:   aimix.ProviderOpenAI,
			Model:      "gpt-other",
			StopReason: aimix.StopReasonToolUse,
			Content: aimix.Contents{
				aimix.ToolCallContent{
					Type:      aimix.ContentTypeToolCall,
					ID:        "call_2|fc_2",
					Name:      "read",
					Arguments: json.RawMessage(`{}`),
				},
			},
		},
		aimix.ToolResultMessage{
			Role:       aimix.RoleToolResult,
			ToolCallID: "call_2|fc_2",
			ToolName:   "read",
		},
		// Another provider: thinking is dropped, the id has no item part.
		aimix.AssistantMessage{
			Role:       aimix.RoleAssistant,
			Api:        aimix.ApiAnthropicMessages,
			Provider:   aimix.ProviderAnthropic,
			Model:      "claude-test",
			StopReason: aimix.StopReasonToolUse,
			Content: aimix.Contents{
				aimix.ThinkingContent{
					Type:              aimix.ContentTypeThinking,
					Thinking:          "hmm",
					ThinkingSignature: "sig",
				},
				aimix.ToolCallContent{
					Type:      aimix.ContentTypeToolCall,
					ID:        "toolu_1",
					Name:      "read",
					Arguments: json.RawMessage(`{}`),
				},
			},
		},
	}
	params, err := buildResponsesParams(model, aimix.TranscriptContext{Messages: messages}, OpenAIResponsesOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var got []map[string]any
	if err := json.Unmarshal([]byte(compactJSON(t, params.Input.OfInputItemList)), &got); err != nil {
		t.Fatal(err)
	}
	want := `[` +
		`{"encrypted_content":"enc1","id":"rs_1","summary":[{"text":"Plan","type":"summary_text"}],"type":"reasoning"},` +
		`{"content":[{"annotations":[],"text":"Hi","type":"output_text"}],"id":"msg_1","phase":"commentary","role":"assistant","status":"completed","type":"message"},` +
		`{"arguments":"{\"path\":\"a.txt\"}","call_id":"call_1","id":"fc_1","name":"read","type":"function_call"},` +
		`{"call_id":"call_1","output":[{"text":"file body","type":"input_text"},{"detail":"auto","image_url":"data:image/png;base64,AAAA","type":"input_image"}],"type":"function_call_output"},` +
		`{"arguments":"{}","call_id":"call_2","name":"read","type":"function_call"},` +
		`{"call_id":"call_2","output":"(no tool output)","type":"function_call_output"},` +
		`{"arguments":"{}","call_id":"toolu_1","name":"read","type":"function_call"},` +
		`{"call_id":"toolu_1","output":"No result provided","type":"function_call_output"}` +
		`]`
	if g := compactJSON(t, got); g != want {
		t.Fatalf("input:\n got %s\nwant %s", g, want)
	}
	if params.Reasoning.Effort != "" || len(params.Include) != 0 {
		t.Fatalf("reasoning sent without effort: %+v", params.Reasoning)
	}
}

func TestResponsesReasoningEffort(t *testing.T) {
	gpt5 := responsesModel("")
	gpt5.ThinkingLevels = aimix.ThinkingLevelMap{
		aimix.ThinkingOff:     "none",
		aimix.ThinkingMinimal: "",
		aimix.ThinkingXhigh:   "xhigh",
	}
	o3 := responsesModel("")
	o3.ThinkingLevels = aimix.ThinkingLevelMap{
		aimix.ThinkingOff:     "",
		aimix.ThinkingMinimal: "",
	}
	plain := responsesModel("")
	plain.Reasoning = false

	tests := []struct {
		name  string
		model aimix.Model
		level aimix.ThinkingLevel
		want  string
	}{
		{
			name:  "off mapped",
			model: gpt5,
			level: aimix.ThinkingOff,
			want:  `{"effort":"none"}`,
		},
		{
			name:  "minimal unsupported goes up",
			model: gpt5,
			level: aimix.ThinkingMinimal,
			want:  `{"effort":"low","summary":"auto"}`,
		},
		{
			name:  "xhigh mapped",
			model: gpt5,
			level: aimix.ThinkingXhigh,
			want:  `{"effort":"xhigh","summary":"auto"}`,
		},
		{
			name:  "max unmapped goes down",
			model: gpt5,
			level: aimix.ThinkingMax,
			want:  `{"effort":"xhigh","summary":"auto"}`,
		},
		{
			name:  "off unsupported goes up",
			model: o3,
			level: aimix.ThinkingOff,
			want:  `{"effort":"low","summary":"auto"}`,
		},
		{
			name:  "off unmapped sends nothing",
			model: responsesModel(""),
			level: aimix.ThinkingOff,
			want:  `{}`,
		},
		{
			name:  "non-reasoning model",
			model: plain,
			level: aimix.ThinkingHigh,
			want:  `{}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			level := effectiveReasoning(tt.model, tt.level)
			params, err := buildResponsesParams(tt.model, aimix.TranscriptContext{}, OpenAIResponsesOptions{
				ReasoningEffort: openAIEffort(tt.model, level),
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := compactJSON(t, params.Reasoning); got != tt.want {
				t.Fatalf("reasoning %s, want %s", got, tt.want)
			}
		})
	}
}

func TestResponsesStopReasons(t *testing.T) {
	created := sseEvent("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","status":"in_progress","model":"gpt-test","output":[]}}`)
	tests := []struct {
		name      string
		status    int
		body      string
		reason    aimix.StopReason
		raw       string
		errReason string
	}{
		{
			name:   "incomplete max tokens",
			status: http.StatusOK,
			body: created + sseEvent("response.incomplete", `{"type":"response.incomplete","sequence_number":1,"response":{"id":"resp_1","status":"incomplete",`+
				`"incomplete_details":{"reason":"max_output_tokens"},"model":"gpt-test","output":[]}}`),
			reason: aimix.StopReasonLength,
			raw:    "incomplete.max_output_tokens",
		},
		{
			name:   "incomplete content filter",
			status: http.StatusOK,
			body: created + sseEvent("response.incomplete", `{"type":"response.incomplete","sequence_number":1,"response":{"id":"resp_1","status":"incomplete",`+
				`"incomplete_details":{"reason":"content_filter"},"model":"gpt-test","output":[]}}`),
			reason:    aimix.StopReasonError,
			raw:       "incomplete.content_filter",
			errReason: "response incomplete: content_filter",
		},
		{
			name:   "failed",
			status: http.StatusOK,
			body: created + sseEvent("response.failed", `{"type":"response.failed","sequence_number":1,"response":{"id":"resp_1","status":"failed",`+
				`"error":{"code":"server_error","message":"boom"},"model":"gpt-test","output":[]}}`),
			reason:    aimix.StopReasonError,
			raw:       "failed",
			errReason: "server_error: boom",
		},
		{
			name:      "error event",
			status:    http.StatusOK,
			body:      created + sseEvent("error", `{"type":"error","sequence_number":1,"code":"rate_limit_exceeded","message":"slow down","param":null}`),
			reason:    aimix.StopReasonError,
			errReason: "OpenAI API error (rate_limit_exceeded): slow down",
		},
		{
			name:      "truncated",
			status:    http.StatusOK,
			body:      created,
			reason:    aimix.StopReasonError,
			errReason: "aimix: OpenAI Responses stream ended before a terminal response event",
		},
		{
			name:      "http error",
			status:    http.StatusUnauthorized,
			body:      `{"error":{"message":"Incorrect API key","type":"invalid_request_error","code":"invalid_api_key","param":null}}`,
			reason:    aimix.StopReasonError,
			errReason: "OpenAI API error (401): Incorrect API key",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := serve(t, tt.status, tt.body)
			s, err := StreamOpenAIResponses(context.Background(), responsesModel(srv.URL), aimix.TranscriptContext{}, OpenAIResponsesOptions{
				StreamOptions: aimix.StreamOptions{
					APIKey: "k",
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			msg, err := aimix.Collect(context.Background(), s)
			if msg.StopReason != tt.reason || msg.RawStopReason != tt.raw || msg.ErrorReason != tt.errReason {
				t.Fatalf("got stop %q raw %q error %q", msg.StopReason, msg.RawStopReason, msg.ErrorReason)
			}
			var se *aimix.StreamError
			if (tt.reason == aimix.StopReasonError) != errors.As(err, &se) {
				t.Fatalf("err = %v", err)
			}
		})
	}
}

func TestNormalizeResponsesToolCallID(t *testing.T) {
	if got := normalizeResponsesToolCallID("toolu_01A.b"); got != "toolu_01A_b" {
		t.Fatalf("got %q", got)
	}
	got := normalizeResponsesToolCallID("call_x|item+1")
	callID, itemID, _ := strings.Cut(got, "|")
	if callID != "call_x" || !strings.HasPrefix(itemID, "fc_") || itemID != foreignItemID("item+1") {
		t.Fatalf("got %q", got)
	}
}

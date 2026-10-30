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

func anthropicModel(baseURL string) aimix.Model {
	return aimix.Model{
		ID:            "claude-test",
		Api:           aimix.ApiAnthropicMessages,
		Provider:      aimix.ProviderAnthropic,
		BaseURL:       baseURL,
		Reasoning:     true,
		ContextWindow: 200000,
		MaxTokens:     64000,
	}
}

var anthropicToolTurn = strings.Join([]string{
	sseEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test-20250101","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":5,"cache_creation_input_tokens":3,"cache_creation":{"ephemeral_1h_input_tokens":2,"ephemeral_5m_input_tokens":1}}}}`),
	sseEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`),
	sseEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}`),
	sseEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig123"}}`),
	sseEvent("content_block_stop", `{"type":"content_block_stop","index":0}`),
	sseEvent("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`),
	sseEvent("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}`),
	sseEvent("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" world"}}`),
	sseEvent("content_block_stop", `{"type":"content_block_stop","index":1}`),
	sseEvent("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"read","input":{}}}`),
	sseEvent("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"pa"}}`),
	sseEvent("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"th\": \"a.txt\"}"}}`),
	sseEvent("content_block_stop", `{"type":"content_block_stop","index":2}`),
	sseEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":42}}`),
	sseEvent("message_stop", `{"type":"message_stop"}`),
}, "")

func TestAnthropicStreamToolTurn(t *testing.T) {
	srv, rec := serve(t, http.StatusOK, anthropicToolTurn)
	model := anthropicModel(srv.URL)
	tc := aimix.NormalizeContext(aimix.Context{
		SystemPrompt: "You are terse.",
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
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
			},
		},
	})
	s, err := StreamSimpleAnthropicMessages(context.Background(), model, tc, aimix.SimpleStreamOptions{
		StreamOptions: aimix.StreamOptions{
			APIKey: "sk-test",
		},
		Reasoning: aimix.ThinkingMedium,
	})
	if err != nil {
		t.Fatal(err)
	}
	events, msg := drain(t, s)

	wantEvents := "start,thinking_start,thinking_delta,thinking_end,text_start,text_delta,text_delta,text_end," +
		"tool_call_start,tool_call_delta,tool_call_delta,tool_call_end,done"
	if got := eventTypes(events); got != wantEvents {
		t.Fatalf("events:\n got %s\nwant %s", got, wantEvents)
	}
	if msg.StopReason != aimix.StopReasonToolUse || msg.RawStopReason != "tool_use" {
		t.Fatalf("stop reason %q (raw %q)", msg.StopReason, msg.RawStopReason)
	}
	if msg.ResponseID != "msg_1" || msg.ResponseModel != "claude-test-20250101" || msg.ThinkingLevel != aimix.ThinkingMedium {
		t.Fatalf("metadata: %+v", msg)
	}
	wantUsage := aimix.Usage{
		Input:        10,
		Output:       42,
		CacheRead:    5,
		CacheWrite:   3,
		CacheWrite1h: 2,
		TotalTokens:  60,
	}
	if msg.Usage != wantUsage {
		t.Fatalf("usage %+v, want %+v", msg.Usage, wantUsage)
	}

	if len(msg.Content) != 3 {
		t.Fatalf("content: %+v", msg.Content)
	}
	th := msg.Content[0].(aimix.ThinkingContent)
	if th.Thinking != "Let me think" || th.ThinkingSignature != "sig123" {
		t.Fatalf("thinking: %+v", th)
	}
	if got := msg.Content[1].(aimix.TextContent).Text; got != "Hello world" {
		t.Fatalf("text: %q", got)
	}
	call := msg.Content[2].(aimix.ToolCallContent)
	if call.ID != "toolu_1" || call.Name != "read" || string(call.Arguments) != `{"path": "a.txt"}` || call.ArgumentsError != "" {
		t.Fatalf("tool call: %+v", call)
	}
	// The first argument delta already shows a parsed partial object.
	if d := events[9].(aimix.StreamToolCallDeltaEvent); string(d.Partial.Content[2].(aimix.ToolCallContent).Arguments) != `{}` {
		t.Fatalf("partial args after first delta: %s", d.Partial.Content[2].(aimix.ToolCallContent).Arguments)
	}
	if end := events[11].(aimix.StreamToolCallEndEvent); end.ToolCall.ID != "toolu_1" {
		t.Fatalf("tool call end: %+v", end.ToolCall)
	}

	body, header, path := rec.get()
	if path != "/v1/messages" {
		t.Fatalf("path %q", path)
	}
	if header.Get("X-Api-Key") != "sk-test" || header.Get("Anthropic-Beta") != interleavedThinkingBeta {
		t.Fatalf("headers: %v", header)
	}
	if body["model"] != "claude-test" || body["stream"] != true {
		t.Fatalf("body: %v", body)
	}
	// Budget thinking: max_tokens is the model cap, the medium budget fits under it.
	if body["max_tokens"] != float64(64000) {
		t.Fatalf("max_tokens %v", body["max_tokens"])
	}
	if got := compactJSON(t, body["thinking"]); got != `{"budget_tokens":8192,"display":"summarized","type":"enabled"}` {
		t.Fatalf("thinking %s", got)
	}
	if got := compactJSON(t, body["system"]); got != `[{"cache_control":{"type":"ephemeral"},"text":"You are terse.","type":"text"}]` {
		t.Fatalf("system %s", got)
	}
	wantTools := `[{"cache_control":{"type":"ephemeral"},"description":"Read a file","eager_input_streaming":true,` +
		`"input_schema":{"properties":{"path":{"type":"string"}},"required":["path"],"type":"object"},"name":"read"}]`
	if got := compactJSON(t, body["tools"]); got != wantTools {
		t.Fatalf("tools:\n got %s\nwant %s", got, wantTools)
	}
	if got := compactJSON(t, body["messages"]); got != `[{"content":[{"cache_control":{"type":"ephemeral"},"text":"read a.txt","type":"text"}],"role":"user"}]` {
		t.Fatalf("messages %s", got)
	}
	if _, ok := body["temperature"]; ok {
		t.Fatal("temperature sent with thinking enabled")
	}
}

func TestAnthropicConvertTranscript(t *testing.T) {
	model := anthropicModel("")
	messages := aimix.Messages{
		aimix.UserMessage{
			Role: aimix.RoleUser,
			Content: aimix.Contents{
				text("look"),
				aimix.ImageContent{
					Type:     aimix.ContentTypeImage,
					Data:     "AAAA",
					MimeType: "image/png",
				},
			},
		},
		// Same model: thinking is replayed with its signature.
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
				aimix.ThinkingContent{
					Type:              aimix.ContentTypeThinking,
					ThinkingSignature: "opaque",
					Redacted:          true,
				},
				aimix.ToolCallContent{
					Type:      aimix.ContentTypeToolCall,
					ID:        "toolu_a",
					Name:      "read",
					Arguments: json.RawMessage(`{"path":"a"}`),
				},
				aimix.ToolCallContent{
					Type:      aimix.ContentTypeToolCall,
					ID:        "toolu_b",
					Name:      "read",
					Arguments: json.RawMessage(`{"path":"b"}`),
				},
			},
		},
		aimix.ToolResultMessage{
			Role:       aimix.RoleToolResult,
			ToolCallID: "toolu_a",
			ToolName:   "read",
			Content: aimix.Contents{
				text("contents of a"),
			},
		},
		// toolu_b has no result: a synthetic error result is inserted.
		// From another provider: thinking is dropped and the tool call id is sanitized.
		aimix.AssistantMessage{
			Role:       aimix.RoleAssistant,
			Api:        aimix.ApiOpenAIResponses,
			Provider:   aimix.ProviderOpenAI,
			Model:      "gpt-test",
			StopReason: aimix.StopReasonToolUse,
			Content: aimix.Contents{
				aimix.ThinkingContent{
					Type:              aimix.ContentTypeThinking,
					Thinking:          "secret",
					ThinkingSignature: `{"type":"reasoning"}`,
				},
				text("calling"),
				aimix.ToolCallContent{
					Type:      aimix.ContentTypeToolCall,
					ID:        "call_1|fc_1",
					Name:      "read",
					Arguments: json.RawMessage(`{"path":"c"}`),
				},
			},
		},
		&aimix.ToolResultMessage{
			Role:       aimix.RoleToolResult,
			ToolCallID: "call_1|fc_1",
			ToolName:   "read",
			IsError:    true,
			Content: aimix.Contents{
				aimix.ImageContent{
					Type:     aimix.ContentTypeImage,
					Data:     "BBBB",
					MimeType: "image/png",
				},
			},
		},
		// Errored turns are not replayed.
		aimix.AssistantMessage{
			Role:       aimix.RoleAssistant,
			StopReason: aimix.StopReasonError,
			Content: aimix.Contents{
				text("partial"),
			},
		},
	}

	params, _, err := buildAnthropicParams(model, aimix.TranscriptContext{Messages: messages}, AnthropicOptions{
		StreamOptions: aimix.StreamOptions{
			CacheRetention: aimix.CacheRetentionNone,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var got []map[string]any
	if err := json.Unmarshal([]byte(compactJSON(t, params.Messages)), &got); err != nil {
		t.Fatal(err)
	}
	want := `[` +
		`{"content":[{"text":"look","type":"text"},{"source":{"data":"AAAA","media_type":"image/png","type":"base64"},"type":"image"}],"role":"user"},` +
		`{"content":[{"signature":"sig","thinking":"hmm","type":"thinking"},{"data":"opaque","type":"redacted_thinking"},` +
		`{"id":"toolu_a","input":{"path":"a"},"name":"read","type":"tool_use"},{"id":"toolu_b","input":{"path":"b"},"name":"read","type":"tool_use"}],"role":"assistant"},` +
		`{"content":[{"content":[{"text":"contents of a","type":"text"}],"is_error":false,"tool_use_id":"toolu_a","type":"tool_result"},` +
		`{"content":[{"text":"No result provided","type":"text"}],"is_error":true,"tool_use_id":"toolu_b","type":"tool_result"}],"role":"user"},` +
		`{"content":[{"text":"calling","type":"text"},{"id":"call_1_fc_1","input":{"path":"c"},"name":"read","type":"tool_use"}],"role":"assistant"},` +
		`{"content":[{"content":[{"text":"(image attached after the tool results)","type":"text"}],"is_error":true,"tool_use_id":"call_1_fc_1","type":"tool_result"},` +
		`{"text":"Images from the error result of tool call call_1_fc_1 (read):","type":"text"},` +
		`{"source":{"data":"BBBB","media_type":"image/png","type":"base64"},"type":"image"}],"role":"user"}` +
		`]`
	if g := compactJSON(t, got); g != want {
		t.Fatalf("messages:\n got %s\nwant %s", g, want)
	}
	if params.Thinking.OfEnabled != nil || params.Thinking.OfDisabled != nil {
		t.Fatalf("thinking sent without ThinkingEnabled: %+v", params.Thinking)
	}
}

func TestAnthropicThinkingOptions(t *testing.T) {
	adaptive := anthropicModel("")
	adaptive.Compat.AdaptiveThinking = true
	adaptive.ThinkingLevels = aimix.ThinkingLevelMap{
		aimix.ThinkingXhigh: "xhigh",
	}

	tests := []struct {
		name     string
		model    aimix.Model
		level    aimix.ThinkingLevel
		thinking string
		effort   string
	}{
		{
			name:     "off",
			model:    anthropicModel(""),
			level:    aimix.ThinkingOff,
			thinking: `{"type":"disabled"}`,
		},
		{
			name:     "adaptive medium",
			model:    adaptive,
			level:    aimix.ThinkingMedium,
			thinking: `{"display":"summarized","type":"adaptive"}`,
			effort:   "medium",
		},
		{
			name:     "adaptive xhigh mapped",
			model:    adaptive,
			level:    aimix.ThinkingXhigh,
			thinking: `{"display":"summarized","type":"adaptive"}`,
			effort:   "xhigh",
		},
		{
			name:     "adaptive max unmapped clamps down",
			model:    adaptive,
			level:    aimix.ThinkingMax,
			thinking: `{"display":"summarized","type":"adaptive"}`,
			effort:   "xhigh",
		},
		{
			name:     "budget xhigh unmapped clamps to high",
			model:    anthropicModel(""),
			level:    aimix.ThinkingXhigh,
			thinking: `{"budget_tokens":16384,"display":"summarized","type":"enabled"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, rec := serve(t, http.StatusOK, anthropicToolTurn)
			tt.model.BaseURL = srv.URL
			s, err := StreamSimpleAnthropicMessages(context.Background(), tt.model, aimix.TranscriptContext{}, aimix.SimpleStreamOptions{
				StreamOptions: aimix.StreamOptions{
					APIKey:      "k",
					Temperature: ptr(0.5),
				},
				Reasoning: tt.level,
			})
			if err != nil {
				t.Fatal(err)
			}
			drain(t, s)
			body, header, _ := rec.get()
			if got := compactJSON(t, body["thinking"]); got != tt.thinking {
				t.Fatalf("thinking %s, want %s", got, tt.thinking)
			}
			var effort string
			if oc, ok := body["output_config"].(map[string]any); ok {
				effort, _ = oc["effort"].(string)
			}
			if effort != tt.effort {
				t.Fatalf("effort %q, want %q", effort, tt.effort)
			}
			_, hasTemp := body["temperature"]
			if hasTemp != (tt.level == aimix.ThinkingOff) {
				t.Fatalf("temperature sent = %v", hasTemp)
			}
			if adaptiveBeta := header.Get("Anthropic-Beta"); tt.model.Compat.AdaptiveThinking && adaptiveBeta != "" {
				t.Fatalf("beta header on adaptive model: %q", adaptiveBeta)
			}
		})
	}
}

func TestAnthropicErrors(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   string
	}{
		{
			name:   "http error",
			status: http.StatusTooManyRequests,
			body:   `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`,
			want:   "Anthropic API error (429): slow down",
		},
		{
			name:   "error event",
			status: http.StatusOK,
			body: sseEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`) +
				sseEvent("error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`),
			want: "Anthropic API error (overloaded_error): Overloaded",
		},
		{
			name:   "truncated",
			status: http.StatusOK,
			body: sseEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`) +
				sseEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
				sseEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"par"}}`),
			want: "aimix: Anthropic stream ended before message_stop",
		},
		{
			name:   "refusal",
			status: http.StatusOK,
			body: sseEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`) +
				sseEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber","explanation":"Not allowed"}},"usage":{"output_tokens":3}}`) +
				sseEvent("message_stop", `{"type":"message_stop"}`),
			want: "Not allowed",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := serve(t, tt.status, tt.body)
			s, err := StreamAnthropicMessages(context.Background(), anthropicModel(srv.URL), aimix.TranscriptContext{}, AnthropicOptions{
				StreamOptions: aimix.StreamOptions{
					APIKey: "k",
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			msg, err := aimix.Collect(context.Background(), s)
			var se *aimix.StreamError
			if !errors.As(err, &se) || se.Reason != aimix.StopReasonError {
				t.Fatalf("err = %v", err)
			}
			if msg.StopReason != aimix.StopReasonError || msg.ErrorReason != tt.want {
				t.Fatalf("stop %q error %q, want %q", msg.StopReason, msg.ErrorReason, tt.want)
			}
		})
	}
}

func TestAnthropicBaseURLWithV1(t *testing.T) {
	for _, suffix := range []string{"/v1", "/v1/", ""} {
		srv, rec := serve(t, http.StatusOK, anthropicToolTurn)
		s, err := StreamAnthropicMessages(context.Background(), anthropicModel(srv.URL+suffix), aimix.TranscriptContext{}, AnthropicOptions{
			StreamOptions: aimix.StreamOptions{
				APIKey: "k",
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		drain(t, s)
		if _, _, path := rec.get(); path != "/v1/messages" {
			t.Fatalf("base URL suffix %q: path %q", suffix, path)
		}
	}
}

func TestAnthropicMissingAPIKey(t *testing.T) {
	_, err := StreamAnthropicMessages(context.Background(), anthropicModel(""), aimix.TranscriptContext{}, AnthropicOptions{})
	if !errors.Is(err, errMissingAPIKey) {
		t.Fatalf("err = %v", err)
	}
}

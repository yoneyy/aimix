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

func completionsModel(baseURL string) aimix.Model {
	return aimix.Model{
		ID:            "deepseek-test",
		Api:           aimix.ApiOpenAICompletions,
		Provider:      "deepseek",
		BaseURL:       baseURL,
		Reasoning:     true,
		ContextWindow: 128000,
		MaxTokens:     8192,
	}
}

func chunk(body string) string {
	return sseEvent("", `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-test-v2",`+body+`}`)
}

var completionsToolTurn = strings.Join([]string{
	chunk(`"choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":"Think"},"finish_reason":null}]`),
	chunk(`"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]`),
	chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"read","arguments":"{\"pa"}}]},"finish_reason":null}]`),
	chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a\"}"}}]},"finish_reason":null}]`),
	chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"read","arguments":"{}"}}]},"finish_reason":null}]`),
	chunk(`"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]`),
	chunk(`"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":10,"total_tokens":60,"prompt_cache_hit_tokens":20,"completion_tokens_details":{"reasoning_tokens":4}}`),
	sseEvent("", "[DONE]"),
}, "")

func TestCompletionsStreamToolTurn(t *testing.T) {
	srv, rec := serve(t, http.StatusOK, completionsToolTurn)
	model := completionsModel(srv.URL)
	model.Compat.LegacyMaxTokens = true
	tc := aimix.NormalizeContext(aimix.Context{
		SystemPrompt: "Be brief.",
		Messages: aimix.Messages{
			aimix.UserMessage{
				Role: aimix.RoleUser,
				Content: aimix.Contents{
					text("read a"),
				},
			},
		},
		Tools: []aimix.Tool{
			{
				Name:       "read",
				Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
			},
		},
	})
	s, err := StreamSimpleOpenAICompletions(context.Background(), model, tc, aimix.SimpleStreamOptions{
		StreamOptions: aimix.StreamOptions{
			APIKey:     "sk-test",
			SessionID:  "s1",
			ToolChoice: aimix.ToolChoiceAuto,
		},
		Reasoning: aimix.ThinkingHigh,
	})
	if err != nil {
		t.Fatal(err)
	}
	events, msg := drain(t, s)

	wantEvents := "start,thinking_start,thinking_delta,text_start,text_delta," +
		"tool_call_start,tool_call_delta,tool_call_delta,tool_call_start,tool_call_delta," +
		"thinking_end,text_end,tool_call_end,tool_call_end,done"
	if got := eventTypes(events); got != wantEvents {
		t.Fatalf("events:\n got %s\nwant %s", got, wantEvents)
	}
	if msg.StopReason != aimix.StopReasonToolUse || msg.RawStopReason != "tool_calls" {
		t.Fatalf("stop reason %q (raw %q)", msg.StopReason, msg.RawStopReason)
	}
	if msg.ResponseID != "chatcmpl-1" || msg.ResponseModel != "deepseek-test-v2" {
		t.Fatalf("metadata: %+v", msg)
	}
	wantUsage := aimix.Usage{
		Input:       30,
		Output:      10,
		CacheRead:   20,
		Reasoning:   4,
		TotalTokens: 60,
	}
	if msg.Usage != wantUsage {
		t.Fatalf("usage %+v, want %+v", msg.Usage, wantUsage)
	}
	th := msg.Content[0].(aimix.ThinkingContent)
	if th.Thinking != "Think" || th.ThinkingSignature != "reasoning_content" {
		t.Fatalf("thinking: %+v", th)
	}
	if got := msg.Content[1].(aimix.TextContent).Text; got != "Hi" {
		t.Fatalf("text %q", got)
	}
	a := msg.Content[2].(aimix.ToolCallContent)
	b := msg.Content[3].(aimix.ToolCallContent)
	if a.ID != "call_a" || string(a.Arguments) != `{"path":"a"}` || b.ID != "call_b" || string(b.Arguments) != `{}` {
		t.Fatalf("tool calls: %+v %+v", a, b)
	}

	body, header, path := rec.get()
	if path != "/chat/completions" || header.Get("Authorization") != "Bearer sk-test" {
		t.Fatalf("path %q, headers %v", path, header)
	}
	checks := map[string]string{
		"stream":           `true`,
		"stream_options":   `{"include_usage":true}`,
		"max_tokens":       `8192`,
		"reasoning_effort": `"high"`,
		"tool_choice":      `"auto"`,
		"messages":         `[{"content":"Be brief.","role":"system"},{"content":"read a","role":"user"}]`,
		"tools":            `[{"function":{"name":"read","parameters":{"properties":{"path":{"type":"string"}},"type":"object"}},"type":"function"}]`,
	}
	for field, want := range checks {
		if got := compactJSON(t, body[field]); got != want {
			t.Errorf("%s:\n got %s\nwant %s", field, got, want)
		}
	}
	// OpenAI-only fields are not sent to other servers.
	for _, field := range []string{"store", "prompt_cache_key", "max_completion_tokens"} {
		if _, ok := body[field]; ok {
			t.Errorf("unexpected field %s", field)
		}
	}
}

func TestCompletionsConvertTranscript(t *testing.T) {
	model := completionsModel("")
	messages := aimix.Messages{
		// Same model: reasoning goes back in reasoning_content.
		aimix.AssistantMessage{
			Role:       aimix.RoleAssistant,
			Api:        aimix.ApiOpenAICompletions,
			Provider:   "deepseek",
			Model:      "deepseek-test",
			StopReason: aimix.StopReasonToolUse,
			Content: aimix.Contents{
				aimix.ThinkingContent{
					Type:              aimix.ContentTypeThinking,
					Thinking:          "Think",
					ThinkingSignature: "reasoning_content",
				},
				aimix.ToolCallContent{
					Type:      aimix.ContentTypeToolCall,
					ID:        "call_a",
					Name:      "read",
					Arguments: json.RawMessage(`{"path":"a"}`),
				},
			},
		},
		aimix.ToolResultMessage{
			Role:       aimix.RoleToolResult,
			ToolCallID: "call_a",
			ToolName:   "read",
			Content: aimix.Contents{
				aimix.ImageContent{
					Type:     aimix.ContentTypeImage,
					Data:     "AAAA",
					MimeType: "image/png",
				},
			},
		},
		// Responses API ids keep both parts.
		aimix.AssistantMessage{
			Role:       aimix.RoleAssistant,
			Api:        aimix.ApiOpenAIResponses,
			Provider:   aimix.ProviderOpenAI,
			Model:      "gpt-test",
			StopReason: aimix.StopReasonToolUse,
			Content: aimix.Contents{
				text("done"),
				aimix.ToolCallContent{
					Type:      aimix.ContentTypeToolCall,
					ID:        "call_1|fc_1",
					Name:      "read",
					Arguments: json.RawMessage(`{}`),
				},
			},
		},
		aimix.ToolResultMessage{
			Role:       aimix.RoleToolResult,
			ToolCallID: "call_1|fc_1",
			ToolName:   "read",
			Content: aimix.Contents{
				text("ok"),
			},
		},
	}
	params, err := buildCompletionsParams(model, aimix.TranscriptContext{Messages: messages}, OpenAICompletionsOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var got []map[string]any
	if err := json.Unmarshal([]byte(compactJSON(t, params.Messages)), &got); err != nil {
		t.Fatal(err)
	}
	want := `[` +
		`{"reasoning_content":"Think","role":"assistant","tool_calls":[{"function":{"arguments":"{\"path\":\"a\"}","name":"read"},"id":"call_a","type":"function"}]},` +
		`{"content":"(see attached image)","role":"tool","tool_call_id":"call_a"},` +
		`{"content":[{"text":"Attached image(s) from tool result:","type":"text"},{"image_url":{"url":"data:image/png;base64,AAAA"},"type":"image_url"}],"role":"user"},` +
		`{"content":"done","role":"assistant","tool_calls":[{"function":{"arguments":"{}","name":"read"},"id":"call_1_fc_1","type":"function"}]},` +
		`{"content":"ok","role":"tool","tool_call_id":"call_1_fc_1"}` +
		`]`
	if g := compactJSON(t, got); g != want {
		t.Fatalf("messages:\n got %s\nwant %s", g, want)
	}
	// Tool calls in the history without declared tools.
	if got := compactJSON(t, params.Tools); got != `[]` {
		t.Fatalf("tools %s", got)
	}
}

func TestCompletionsOpenAIFields(t *testing.T) {
	model := completionsModel("")
	model.Provider = aimix.ProviderOpenAI
	params, err := buildCompletionsParams(model, aimix.TranscriptContext{}, OpenAICompletionsOptions{
		StreamOptions: aimix.StreamOptions{
			MaxTokens:      100,
			SessionID:      "s1",
			CacheRetention: aimix.CacheRetentionLong,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	_ = json.Unmarshal([]byte(compactJSON(t, params)), &body)
	if body["store"] != false || body["prompt_cache_key"] != "s1" || body["prompt_cache_retention"] != "24h" || body["max_completion_tokens"] != float64(100) {
		t.Fatalf("body %v", body)
	}
}

func TestNormalizeCompletionsToolCallID(t *testing.T) {
	long := "call_" + strings.Repeat("x", 30) + "|fc_" + strings.Repeat("y", 30)
	tests := []struct {
		in, want string
	}{
		{in: "call_1|fc_1", want: "call_1_fc_1"},
		{in: "call+1|fc/2", want: "call_1_fc_2"},
		{in: strings.Repeat("z", 50), want: strings.Repeat("z", 40)},
	}
	for _, tt := range tests {
		if got := normalizeCompletionsToolCallID(tt.in); got != tt.want {
			t.Errorf("%q: got %q, want %q", tt.in, got, tt.want)
		}
	}
	if got := normalizeCompletionsToolCallID(long); len(got) > 40 || !strings.HasPrefix(got, "call_xxx") {
		t.Errorf("long id: got %q", got)
	}
}

func TestCompletionsErrors(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "content filter",
			body: chunk(`"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"content_filter"}]`) + sseEvent("", "[DONE]"),
			want: "provider finish_reason: content_filter",
		},
		{
			name: "no finish reason",
			body: chunk(`"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]`) + sseEvent("", "[DONE]"),
			want: "aimix: Chat Completions stream ended without finish_reason",
		},
		{
			name: "error chunk",
			body: chunk(`"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]`) +
				sseEvent("", `{"error":{"message":"upstream overloaded","type":"server_error"}}`),
			want: "OpenAI API error: upstream overloaded",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := serve(t, http.StatusOK, tt.body)
			s, err := StreamOpenAICompletions(context.Background(), completionsModel(srv.URL), aimix.TranscriptContext{}, OpenAICompletionsOptions{
				StreamOptions: aimix.StreamOptions{
					APIKey: "k",
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			msg, err := aimix.Collect(context.Background(), s)
			var se *aimix.StreamError
			if !errors.As(err, &se) || msg.ErrorReason != tt.want {
				t.Fatalf("err %v, error reason %q, want %q", err, msg.ErrorReason, tt.want)
			}
			// Partial content survives the error.
			if len(msg.Content) == 0 {
				t.Fatal("partial content lost")
			}
		})
	}
}

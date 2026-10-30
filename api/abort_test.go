package api

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/yoneyy/aimix"
)

// TestAbort cancels the context in the middle of a stream: the turn ends with
// StopReasonAborted and keeps the text streamed so far.
func TestAbort(t *testing.T) {
	tests := []struct {
		name   string
		prefix string
		stream func(ctx context.Context, baseURL string) (*aimix.AssistantMessageEventStream, error)
	}{
		{
			name: "anthropic",
			prefix: sseEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`) +
				sseEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
				sseEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`),
			stream: func(ctx context.Context, baseURL string) (*aimix.AssistantMessageEventStream, error) {
				return StreamAnthropicMessages(ctx, anthropicModel(baseURL), aimix.TranscriptContext{}, AnthropicOptions{
					StreamOptions: aimix.StreamOptions{
						APIKey: "k",
					},
				})
			},
		},
		{
			name: "responses",
			prefix: sseEvent("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}`) +
				sseEvent("response.output_text.delta", `{"type":"response.output_text.delta","sequence_number":2,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"partial","logprobs":[]}`),
			stream: func(ctx context.Context, baseURL string) (*aimix.AssistantMessageEventStream, error) {
				return StreamOpenAIResponses(ctx, responsesModel(baseURL), aimix.TranscriptContext{}, OpenAIResponsesOptions{
					StreamOptions: aimix.StreamOptions{
						APIKey: "k",
					},
				})
			},
		},
		{
			name:   "completions",
			prefix: chunk(`"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]`),
			stream: func(ctx context.Context, baseURL string) (*aimix.AssistantMessageEventStream, error) {
				return StreamOpenAICompletions(ctx, completionsModel(baseURL), aimix.TranscriptContext{}, OpenAICompletionsOptions{
					StreamOptions: aimix.StreamOptions{
						APIKey: "k",
					},
				})
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// The server sends the prefix and then holds the stream open.
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, tt.prefix)
				w.(http.Flusher).Flush()
				<-r.Context().Done()
			}))
			defer srv.Close()

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			s, err := tt.stream(ctx, srv.URL)
			if err != nil {
				t.Fatal(err)
			}
			for ev := range s.Events() {
				if ev.Event() == aimix.StreamEventTextDelta {
					cancel()
				}
			}
			msg, err := s.Result(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if msg.StopReason != aimix.StopReasonAborted {
				t.Fatalf("stop reason %q (%s)", msg.StopReason, msg.ErrorReason)
			}
			if len(msg.Content) != 1 || msg.Content[0].(aimix.TextContent).Text != "partial" {
				t.Fatalf("content %+v", msg.Content)
			}

			_, err = aimix.Collect(context.Background(), s)
			var se *aimix.StreamError
			if !errors.As(err, &se) || se.Reason != aimix.StopReasonAborted {
				t.Fatalf("Collect err = %v", err)
			}
		})
	}
}

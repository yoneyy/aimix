package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yoneyy/aimix"
)

// recorded is the request a test server received.
type recorded struct {
	mu     sync.Mutex
	body   map[string]any
	header http.Header
	path   string
}

func (r *recorded) get() (map[string]any, http.Header, string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.body, r.header, r.path
}

// serve starts a server that records the request and answers with payload: an
// SSE stream when status is 200, a JSON error body otherwise.
func serve(t *testing.T, status int, payload string) (*httptest.Server, *recorded) {
	t.Helper()
	rec := &recorded{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		rec.mu.Lock()
		_ = json.Unmarshal(raw, &rec.body)
		rec.header = r.Header.Clone()
		rec.path = r.URL.Path
		rec.mu.Unlock()
		if status != http.StatusOK {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			_, _ = io.WriteString(w, payload)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, payload)
	}))
	t.Cleanup(srv.Close)
	return srv, rec
}

// sseEvent renders one SSE frame. An empty name renders a data-only frame.
func sseEvent(name, data string) string {
	if name == "" {
		return fmt.Sprintf("data: %s\n\n", data)
	}
	return fmt.Sprintf("event: %s\ndata: %s\n\n", name, data)
}

// drain reads every event and the final message.
func drain(t *testing.T, s *aimix.AssistantMessageEventStream) ([]aimix.AssistantMessageEvent, aimix.AssistantMessage) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	var events []aimix.AssistantMessageEvent
	for ev := range s.Events() {
		events = append(events, ev)
	}
	msg, err := s.Result(ctx)
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	return events, msg
}

func eventTypes(events []aimix.AssistantMessageEvent) string {
	names := make([]string, len(events))
	for i, ev := range events {
		names[i] = string(ev.Event())
	}
	return strings.Join(names, ",")
}

// compactJSON renders v as compact JSON for comparisons.
func compactJSON(t *testing.T, v any) string {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(raw)
}

func text(s string) aimix.TextContent {
	return aimix.TextContent{
		Type: aimix.ContentTypeText,
		Text: s,
	}
}

func ptr[T any](v T) *T {
	return &v
}

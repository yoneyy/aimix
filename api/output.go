package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/yoneyy/aimix"
	"github.com/yoneyy/aimix/infra"
)

// eventBufferSize lets the producer run a little ahead of the consumer.
const eventBufferSize = 32

// streamWriter builds the assistant message of one response and emits the events
// describing it. Every event carries a snapshot of the message, so the message is
// never shared with consumers while it is being written.
type streamWriter struct {
	ctx    context.Context
	stream *aimix.AssistantMessageEventStream
	out    aimix.AssistantMessage

	texts map[int]*strings.Builder   // text and thinking blocks, by content index
	sigs  map[int]*strings.Builder   // thinking signatures that arrive in pieces
	args  map[int]*infra.PartialJSON // tool call arguments
}

func newStreamWriter(ctx context.Context, model aimix.Model) *streamWriter {
	return &streamWriter{
		ctx:    ctx,
		stream: aimix.NewAssistantMessageEventStream(eventBufferSize),
		out: aimix.AssistantMessage{
			Role:      aimix.RoleAssistant,
			Content:   aimix.Contents{},
			Api:       model.Api,
			Provider:  model.Provider,
			Model:     model.ID,
			Timestamp: time.Now().UnixMilli(),
		},
		texts: make(map[int]*strings.Builder),
		sigs:  make(map[int]*strings.Builder),
		args:  make(map[int]*infra.PartialJSON),
	}
}

// run calls body in a new goroutine and ends the stream with a done event, or with
// an error event when body fails, panics or the context is canceled. body must set
// out.StopReason.
func (w *streamWriter) run(body func() error) *aimix.AssistantMessageEventStream {
	go func() {
		defer w.stream.Close()
		err := w.call(body)
		if err == nil && w.ctx.Err() != nil {
			err = w.ctx.Err()
		}
		if err == nil && w.out.StopReason == "" {
			err = errors.New("aimix: stream ended without a stop reason")
		}
		if err != nil {
			w.fail(err)
			return
		}
		w.done()
	}()
	return w.stream
}

func (w *streamWriter) call(body func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("aimix: panic while streaming: %v", r)
		}
	}()
	return body()
}

func (w *streamWriter) emit(ev aimix.AssistantMessageEvent) {
	// A canceled context drops the event; body notices the cancellation through
	// the SDK stream and run reports it.
	_ = w.stream.Emit(w.ctx, ev)
}

func (w *streamWriter) snapshot() aimix.AssistantMessage {
	m := w.out
	m.Content = slices.Clone(w.out.Content)
	return m
}

func (w *streamWriter) done() {
	if w.out.StopReason == aimix.StopReasonStop && w.hasToolCalls() {
		w.out.StopReason = aimix.StopReasonToolUse
	}
	w.emit(aimix.StreamDoneEvent{
		Type:    aimix.StreamEventDone,
		Reason:  w.out.StopReason,
		Message: w.snapshot(),
	})
}

func (w *streamWriter) fail(err error) {
	reason := aimix.StopReasonError
	if w.ctx.Err() != nil || errors.Is(err, context.Canceled) {
		reason = aimix.StopReasonAborted
	}
	w.out.StopReason = reason
	w.out.ErrorReason = formatError(err)
	w.emit(aimix.StreamErrorEvent{
		Type:    aimix.StreamEventError,
		Reason:  reason,
		Message: w.snapshot(),
		Err:     err,
	})
}

func (w *streamWriter) hasToolCalls() bool {
	for _, c := range w.out.Content {
		if _, ok := c.(aimix.ToolCallContent); ok {
			return true
		}
	}
	return false
}

func (w *streamWriter) start() {
	w.emit(aimix.StreamStartEvent{
		Type:    aimix.StreamEventStart,
		Partial: w.snapshot(),
	})
}

// update replaces content block i with f applied to it.
func update[T aimix.Content](w *streamWriter, i int, f func(*T)) {
	block := w.out.Content[i].(T)
	f(&block)
	w.out.Content[i] = block
}

func (w *streamWriter) startText() int {
	i := len(w.out.Content)
	w.out.Content = append(w.out.Content, aimix.TextContent{
		Type: aimix.ContentTypeText,
	})
	w.texts[i] = &strings.Builder{}
	w.emit(aimix.StreamTextStartEvent{
		Type:         aimix.StreamEventTextStart,
		ContentIndex: i,
		Partial:      w.snapshot(),
	})
	return i
}

func (w *streamWriter) appendText(i int, delta string) {
	if delta == "" {
		return
	}
	b := w.texts[i]
	b.WriteString(delta)
	update(w, i, func(c *aimix.TextContent) {
		c.Text = b.String()
	})
	w.emit(aimix.StreamTextDeltaEvent{
		Type:         aimix.StreamEventTextDelta,
		ContentIndex: i,
		Delta:        delta,
		Partial:      w.snapshot(),
	})
}

func (w *streamWriter) setTextSignature(i int, sig string) {
	update(w, i, func(c *aimix.TextContent) {
		c.TextSignature = sig
	})
}

func (w *streamWriter) endText(i int) {
	w.emit(aimix.StreamTextEndEvent{
		Type:         aimix.StreamEventTextEnd,
		ContentIndex: i,
		Content:      w.out.Content[i].(aimix.TextContent).Text,
		Partial:      w.snapshot(),
	})
}

func (w *streamWriter) startThinking() int {
	i := len(w.out.Content)
	w.out.Content = append(w.out.Content, aimix.ThinkingContent{
		Type: aimix.ContentTypeThinking,
	})
	w.texts[i] = &strings.Builder{}
	w.sigs[i] = &strings.Builder{}
	w.emit(aimix.StreamThinkingStartEvent{
		Type:         aimix.StreamEventThinkingStart,
		ContentIndex: i,
		Partial:      w.snapshot(),
	})
	return i
}

func (w *streamWriter) appendThinking(i int, delta string) {
	if delta == "" {
		return
	}
	b := w.texts[i]
	b.WriteString(delta)
	update(w, i, func(c *aimix.ThinkingContent) {
		c.Thinking = b.String()
	})
	w.emit(aimix.StreamThinkingDeltaEvent{
		Type:         aimix.StreamEventThinkingDelta,
		ContentIndex: i,
		Delta:        delta,
		Partial:      w.snapshot(),
	})
}

// appendThinkingSignature extends the signature without emitting an event.
func (w *streamWriter) appendThinkingSignature(i int, delta string) {
	b := w.sigs[i]
	b.WriteString(delta)
	update(w, i, func(c *aimix.ThinkingContent) {
		c.ThinkingSignature = b.String()
	})
}

// setThinking replaces the thinking text and signature once the block is final.
func (w *streamWriter) setThinking(i int, f func(*aimix.ThinkingContent)) {
	update(w, i, f)
}

func (w *streamWriter) endThinking(i int) {
	w.emit(aimix.StreamThinkingEndEvent{
		Type:         aimix.StreamEventThinkingEnd,
		ContentIndex: i,
		Content:      w.out.Content[i].(aimix.ThinkingContent).Thinking,
		Partial:      w.snapshot(),
	})
}

func (w *streamWriter) startToolCall(id, name string) int {
	i := len(w.out.Content)
	w.out.Content = append(w.out.Content, aimix.ToolCallContent{
		Type:      aimix.ContentTypeToolCall,
		ID:        id,
		Name:      name,
		Arguments: json.RawMessage("{}"),
	})
	w.args[i] = &infra.PartialJSON{}
	w.emit(aimix.StreamToolCallStartEvent{
		Type:         aimix.StreamEventToolCallStart,
		ContentIndex: i,
		Partial:      w.snapshot(),
	})
	return i
}

func (w *streamWriter) updateToolCall(i int, f func(*aimix.ToolCallContent)) {
	update(w, i, f)
}

// appendToolArgs extends the arguments text and shows the partial arguments
// parsed so far.
func (w *streamWriter) appendToolArgs(i int, delta string) {
	if delta == "" {
		return
	}
	p := w.args[i]
	p.Write(delta)
	if v, ok := p.Value(); ok && len(v) > 0 && v[0] == '{' {
		update(w, i, func(c *aimix.ToolCallContent) {
			c.Arguments = v
		})
	}
	w.emit(aimix.StreamToolCallDeltaEvent{
		Type:         aimix.StreamEventToolCallDelta,
		ContentIndex: i,
		Delta:        delta,
		Partial:      w.snapshot(),
	})
}

// replaceToolArgs sets the complete arguments text when the API resends it,
// emitting any part that was not streamed as a delta.
func (w *streamWriter) replaceToolArgs(i int, full string) {
	p := w.args[i]
	seen := p.Text()
	if full == seen {
		return
	}
	if strings.HasPrefix(full, seen) {
		w.appendToolArgs(i, full[len(seen):])
		return
	}
	p.Reset()
	w.appendToolArgs(i, full)
}

// endToolCall parses the complete arguments. Text that is not a JSON object
// leaves the arguments empty and sets ArgumentsError.
func (w *streamWriter) endToolCall(i int) {
	text := w.args[i].Text()
	args, err := infra.ParseJSONObject(text)
	update(w, i, func(c *aimix.ToolCallContent) {
		if err != nil {
			c.Arguments = json.RawMessage("{}")
			c.ArgumentsError = fmt.Sprintf("%v\n\nReceived arguments:\n%s", err, text)
			return
		}
		c.Arguments = args
	})
	w.emit(aimix.StreamToolCallEndEvent{
		Type:         aimix.StreamEventToolCallEnd,
		ContentIndex: i,
		ToolCall:     w.out.Content[i].(aimix.ToolCallContent),
		Partial:      w.snapshot(),
	})
}

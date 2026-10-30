package aimix

import "github.com/yoneyy/aimix/infra"

type StreamEvent string

const (
	StreamEventStart         StreamEvent = "start"
	StreamEventTextStart     StreamEvent = "text_start"
	StreamEventTextDelta     StreamEvent = "text_delta"
	StreamEventTextEnd       StreamEvent = "text_end"
	StreamEventThinkingStart StreamEvent = "thinking_start"
	StreamEventThinkingDelta StreamEvent = "thinking_delta"
	StreamEventThinkingEnd   StreamEvent = "thinking_end"
	StreamEventToolCallStart StreamEvent = "tool_call_start"
	StreamEventToolCallDelta StreamEvent = "tool_call_delta"
	StreamEventToolCallEnd   StreamEvent = "tool_call_end"
	StreamEventDone          StreamEvent = "done"
	StreamEventError         StreamEvent = "error"
)

// AssistantMessageEvent is one event of an assistant response stream. Every event
// carries Partial, a snapshot of the message built so far: it is never modified
// after the event is emitted, so consumers may keep it.
type AssistantMessageEvent interface {
	implAssistantMessageEvent()
	Event() StreamEvent
}

type StreamStartEvent struct {
	Type    StreamEvent
	Partial AssistantMessage
}

type StreamTextStartEvent struct {
	Type         StreamEvent
	ContentIndex int
	Partial      AssistantMessage
}

type StreamTextDeltaEvent struct {
	Type         StreamEvent
	ContentIndex int
	Delta        string
	Partial      AssistantMessage
}

type StreamTextEndEvent struct {
	Type         StreamEvent
	ContentIndex int
	Content      string
	Partial      AssistantMessage
}

type StreamThinkingStartEvent struct {
	Type         StreamEvent
	ContentIndex int
	Partial      AssistantMessage
}

type StreamThinkingDeltaEvent struct {
	Type         StreamEvent
	ContentIndex int
	Delta        string
	Partial      AssistantMessage
}

type StreamThinkingEndEvent struct {
	Type         StreamEvent
	ContentIndex int
	Content      string
	Partial      AssistantMessage
}

type StreamToolCallStartEvent struct {
	Type         StreamEvent
	ContentIndex int
	Partial      AssistantMessage
}

type StreamToolCallDeltaEvent struct {
	Type         StreamEvent
	ContentIndex int
	Delta        string
	Partial      AssistantMessage
}

type StreamToolCallEndEvent struct {
	Type         StreamEvent
	ContentIndex int
	ToolCall     ToolCallContent
	Partial      AssistantMessage
}

type StreamDoneEvent struct {
	Type    StreamEvent
	Reason  StopReason
	Message AssistantMessage
}

type StreamErrorEvent struct {
	Type    StreamEvent
	Reason  StopReason
	Message AssistantMessage
	Err     error
}

// impl AssistantMessageEvent
func (StreamStartEvent) implAssistantMessageEvent()         {}
func (StreamTextStartEvent) implAssistantMessageEvent()     {}
func (StreamTextDeltaEvent) implAssistantMessageEvent()     {}
func (StreamTextEndEvent) implAssistantMessageEvent()       {}
func (StreamThinkingStartEvent) implAssistantMessageEvent() {}
func (StreamThinkingDeltaEvent) implAssistantMessageEvent() {}
func (StreamThinkingEndEvent) implAssistantMessageEvent()   {}
func (StreamToolCallStartEvent) implAssistantMessageEvent() {}
func (StreamToolCallDeltaEvent) implAssistantMessageEvent() {}
func (StreamToolCallEndEvent) implAssistantMessageEvent()   {}
func (StreamDoneEvent) implAssistantMessageEvent()          {}
func (StreamErrorEvent) implAssistantMessageEvent()         {}

func (e StreamStartEvent) Event() StreamEvent         { return StreamEventStart }
func (e StreamTextStartEvent) Event() StreamEvent     { return StreamEventTextStart }
func (e StreamTextDeltaEvent) Event() StreamEvent     { return StreamEventTextDelta }
func (e StreamTextEndEvent) Event() StreamEvent       { return StreamEventTextEnd }
func (e StreamThinkingStartEvent) Event() StreamEvent { return StreamEventThinkingStart }
func (e StreamThinkingDeltaEvent) Event() StreamEvent { return StreamEventThinkingDelta }
func (e StreamThinkingEndEvent) Event() StreamEvent   { return StreamEventThinkingEnd }
func (e StreamToolCallStartEvent) Event() StreamEvent { return StreamEventToolCallStart }
func (e StreamToolCallDeltaEvent) Event() StreamEvent { return StreamEventToolCallDelta }
func (e StreamToolCallEndEvent) Event() StreamEvent   { return StreamEventToolCallEnd }
func (e StreamDoneEvent) Event() StreamEvent          { return StreamEventDone }
func (e StreamErrorEvent) Event() StreamEvent         { return StreamEventError }

type AssistantMessageEventStream = infra.EventStream[AssistantMessageEvent, AssistantMessage]

func NewAssistantMessageEventStream(bufSize int) *AssistantMessageEventStream {
	es := infra.NewEventStream[AssistantMessageEvent, AssistantMessage](bufSize)
	es.IsComplete = func(event AssistantMessageEvent) bool {
		e := event.Event()
		return e == StreamEventDone || e == StreamEventError
	}
	es.ExtractResult = func(event AssistantMessageEvent) AssistantMessage {
		switch t := event.(type) {
		case StreamDoneEvent:
			return t.Message
		case StreamErrorEvent:
			return t.Message
		default:
			return AssistantMessage{}
		}
	}
	return es
}

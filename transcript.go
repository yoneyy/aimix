package aimix

import "strings"

type Role string

func (r Role) String() string {
	return string(r)
}

const (
	RoleSystem     Role = "system"
	RoleUser       Role = "user"
	RoleAssistant  Role = "assistant"
	RoleToolResult Role = "toolResult"
)

// Messages is a transcript. It has its own type so that it can be decoded from
// JSON into the concrete message types.
type Messages []Message

// TranscriptContext is the provider-facing form of a Context: the system prompt and
// tools live in system messages inside the transcript. NormalizeContext is the only
// way to build one from a Context.
type TranscriptContext struct {
	Messages Messages `json:"messages"`
}

// NormalizeContext folds Context.SystemPrompt and Context.Tools into a leading system
// message. An empty prompt with no tools adds nothing.
func NormalizeContext(c Context) TranscriptContext {
	if c.SystemPrompt == "" && len(c.Tools) == 0 {
		return TranscriptContext{Messages: c.Messages}
	}
	head := SystemMessage{
		Role:       RoleSystem,
		ToolsAdded: c.Tools,
	}
	if c.SystemPrompt != "" {
		head.Content = []TextContent{
			{
				Type: ContentTypeText,
				Text: c.SystemPrompt,
			},
		}
	}
	messages := make(Messages, 0, len(c.Messages)+1)
	messages = append(messages, head)
	messages = append(messages, c.Messages...)
	return TranscriptContext{Messages: messages}
}

// CurrentTools resolves the tools available after applying every system message's
// ToolsRemoved and ToolsAdded in order. A re-added tool keeps its original position.
func CurrentTools(messages []Message) []Tool {
	var tools []Tool
	for _, m := range messages {
		sm, ok := AsSystemMessage(m)
		if !ok {
			continue
		}
		for _, name := range sm.ToolsRemoved {
			for i, t := range tools {
				if t.Name == name {
					tools = append(tools[:i], tools[i+1:]...)
					break
				}
			}
		}
		for _, added := range sm.ToolsAdded {
			replaced := false
			for i, t := range tools {
				if t.Name == added.Name {
					tools[i] = added
					replaced = true
					break
				}
			}
			if !replaced {
				tools = append(tools, added)
			}
		}
	}
	return tools
}

// SystemPromptText joins the text of every system message, in order, with blank lines.
func SystemPromptText(messages []Message) string {
	var parts []string
	for _, m := range messages {
		sm, ok := AsSystemMessage(m)
		if !ok {
			continue
		}
		if text := TextOf(sm.Content); text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, "\n\n")
}

// CollapseSystemMessages rebuilds the transcript for APIs without mid-conversation
// system messages: one leading system message holds the replayed prompt and the
// current tools, and every other system message is dropped.
func CollapseSystemMessages(tc TranscriptContext) TranscriptContext {
	var (
		hasSystem bool
		timestamp int64
		rest      = make(Messages, 0, len(tc.Messages))
	)
	for _, m := range tc.Messages {
		if sm, ok := AsSystemMessage(m); ok {
			if !hasSystem {
				timestamp = sm.Timestamp
			}
			hasSystem = true
			continue
		}
		rest = append(rest, m)
	}
	if !hasSystem {
		return TranscriptContext{Messages: rest}
	}

	head := SystemMessage{
		Role:       RoleSystem,
		ToolsAdded: CurrentTools(tc.Messages),
		Timestamp:  timestamp,
	}
	if text := SystemPromptText(tc.Messages); text != "" {
		head.Content = []TextContent{
			{
				Type: ContentTypeText,
				Text: text,
			},
		}
	}
	return TranscriptContext{Messages: append(Messages{head}, rest...)}
}

// AsSystemMessage reports whether m is a system message, accepting both the value
// and the pointer form.
func AsSystemMessage(m Message) (SystemMessage, bool) {
	switch v := m.(type) {
	case SystemMessage:
		return v, true
	case *SystemMessage:
		if v != nil {
			return *v, true
		}
	}
	return SystemMessage{}, false
}

// TextOf joins the text of the given text blocks with newlines.
func TextOf(blocks []TextContent) string {
	switch len(blocks) {
	case 0:
		return ""
	case 1:
		return blocks[0].Text
	}
	texts := make([]string, len(blocks))
	for i, b := range blocks {
		texts[i] = b.Text
	}
	return strings.Join(texts, "\n")
}

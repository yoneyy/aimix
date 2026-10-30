package api

import (
	"strings"
	"time"

	"github.com/yoneyy/aimix"
)

const (
	nonVisionUserImagePlaceholder = "(image omitted: model does not support images)"
	nonVisionToolImagePlaceholder = "(tool image omitted: model does not support images)"
)

// transformMessages prepares a transcript for model:
//   - pointer messages and content blocks become values, so converters only
//     need to handle the value types;
//   - images become placeholders when the model does not accept them;
//   - assistant turns from another provider, api or model lose their thinking and
//     signatures, and their tool call ids go through normalizeID (nil keeps them),
//     with the matching tool results rewritten;
//   - errored and aborted assistant turns are dropped;
//   - tool calls without a result get an error result "No result provided".
func transformMessages(messages []aimix.Message, model aimix.Model, normalizeID func(string) string) []aimix.Message {
	values := make([]aimix.Message, 0, len(messages))
	for _, m := range messages {
		if v := messageValue(m); v != nil {
			values = append(values, v)
		}
	}
	if !model.SupportsImages() {
		values = downgradeImages(values)
	}

	// First pass: rewrite assistant turns and the tool results that answer them.
	idMap := make(map[string]string)
	for i, m := range values {
		switch v := m.(type) {
		case aimix.ToolResultMessage:
			if id, ok := idMap[v.ToolCallID]; ok {
				v.ToolCallID = id
				values[i] = v
			}
		case aimix.AssistantMessage:
			sameModel := v.Provider == model.Provider && v.Api == model.Api && v.Model == model.ID
			content := make(aimix.Contents, 0, len(v.Content))
			for _, c := range v.Content {
				switch b := c.(type) {
				case aimix.ThinkingContent:
					// Signatures and encrypted reasoning only work for the model that
					// produced them, and readable thinking sent as text would look like
					// part of the visible reply.
					if !sameModel {
						continue
					}
					if !b.Redacted && b.ThinkingSignature == "" && strings.TrimSpace(b.Thinking) == "" {
						continue
					}
					content = append(content, b)
				case aimix.TextContent:
					if !sameModel {
						b.TextSignature = ""
					}
					content = append(content, b)
				case aimix.ToolCallContent:
					if !sameModel {
						b.ThoughtSignature = ""
						if normalizeID != nil {
							if id := normalizeID(b.ID); id != b.ID {
								idMap[b.ID] = id
								b.ID = id
							}
						}
					}
					content = append(content, b)
				default:
					content = append(content, c)
				}
			}
			v.Content = content
			values[i] = v
		}
	}

	// Second pass: drop failed turns and close tool calls left without results.
	// A system message between a tool call and its results is held back until
	// after the results.
	var (
		result  = make([]aimix.Message, 0, len(values))
		pending []aimix.ToolCallContent
		seen    = make(map[string]bool)
		held    []aimix.Message
	)
	closePending := func() {
		for _, tc := range pending {
			if seen[tc.ID] {
				continue
			}
			result = append(result, aimix.ToolResultMessage{
				Role:       aimix.RoleToolResult,
				ToolCallID: tc.ID,
				ToolName:   tc.Name,
				Content: aimix.Contents{
					aimix.TextContent{
						Type: aimix.ContentTypeText,
						Text: "No result provided",
					},
				},
				IsError:   true,
				Timestamp: time.Now().UnixMilli(),
			})
		}
		pending = nil
		clear(seen)
		result = append(result, held...)
		held = nil
	}
	for _, m := range values {
		switch v := m.(type) {
		case aimix.AssistantMessage:
			closePending()
			if v.StopReason == aimix.StopReasonError || v.StopReason == aimix.StopReasonAborted {
				continue
			}
			for _, c := range v.Content {
				if tc, ok := c.(aimix.ToolCallContent); ok {
					pending = append(pending, tc)
				}
			}
			result = append(result, v)
		case aimix.ToolResultMessage:
			seen[v.ToolCallID] = true
			result = append(result, v)
		case aimix.SystemMessage:
			if len(pending) > 0 {
				held = append(held, v)
			} else {
				result = append(result, v)
			}
		default:
			closePending()
			result = append(result, v)
		}
	}
	closePending()
	return result
}

// messageValue returns the value form of m with value content blocks, or nil
// for a nil pointer.
func messageValue(m aimix.Message) aimix.Message {
	switch v := m.(type) {
	case *aimix.SystemMessage:
		if v == nil {
			return nil
		}
		return *v
	case *aimix.UserMessage:
		if v == nil {
			return nil
		}
		return messageValue(*v)
	case *aimix.AssistantMessage:
		if v == nil {
			return nil
		}
		return messageValue(*v)
	case *aimix.ToolResultMessage:
		if v == nil {
			return nil
		}
		return messageValue(*v)
	case aimix.UserMessage:
		v.Content = contentValues(v.Content)
		return v
	case aimix.AssistantMessage:
		v.Content = contentValues(v.Content)
		return v
	case aimix.ToolResultMessage:
		v.Content = contentValues(v.Content)
		return v
	}
	return m
}

func contentValues(cs aimix.Contents) aimix.Contents {
	out := make(aimix.Contents, 0, len(cs))
	for _, c := range cs {
		switch v := c.(type) {
		case *aimix.TextContent:
			if v != nil {
				out = append(out, *v)
			}
		case *aimix.ThinkingContent:
			if v != nil {
				out = append(out, *v)
			}
		case *aimix.ImageContent:
			if v != nil {
				out = append(out, *v)
			}
		case *aimix.ToolCallContent:
			if v != nil {
				out = append(out, *v)
			}
		case nil:
		default:
			out = append(out, c)
		}
	}
	return out
}

func downgradeImages(messages []aimix.Message) []aimix.Message {
	for i, m := range messages {
		switch v := m.(type) {
		case aimix.UserMessage:
			v.Content = replaceImages(v.Content, nonVisionUserImagePlaceholder)
			messages[i] = v
		case aimix.ToolResultMessage:
			v.Content = replaceImages(v.Content, nonVisionToolImagePlaceholder)
			messages[i] = v
		}
	}
	return messages
}

// replaceImages replaces each run of images with one placeholder text block.
func replaceImages(cs aimix.Contents, placeholder string) aimix.Contents {
	out := make(aimix.Contents, 0, len(cs))
	previousWasPlaceholder := false
	for _, c := range cs {
		if _, ok := c.(aimix.ImageContent); ok {
			if !previousWasPlaceholder {
				out = append(out, aimix.TextContent{
					Type: aimix.ContentTypeText,
					Text: placeholder,
				})
			}
			previousWasPlaceholder = true
			continue
		}
		out = append(out, c)
		tc, ok := c.(aimix.TextContent)
		previousWasPlaceholder = ok && tc.Text == placeholder
	}
	return out
}

// splitSystem separates the leading system message of a collapsed transcript.
func splitSystem(messages []aimix.Message) (system aimix.SystemMessage, rest []aimix.Message) {
	if len(messages) > 0 {
		if sm, ok := aimix.AsSystemMessage(messages[0]); ok {
			return sm, messages[1:]
		}
	}
	return aimix.SystemMessage{}, messages
}

// sanitizeID keeps [a-zA-Z0-9_-], replacing anything else with '_', and cuts the
// result to maxLen bytes.
func sanitizeID(id string, maxLen int) string {
	b := []byte(id)
	for i, c := range b {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
			b[i] = '_'
		}
	}
	if len(b) > maxLen {
		b = b[:maxLen]
	}
	return string(b)
}

// textBlocks returns the text of the text blocks in cs.
func textBlocks(cs aimix.Contents) []string {
	var texts []string
	for _, c := range cs {
		if tc, ok := c.(aimix.TextContent); ok {
			texts = append(texts, tc.Text)
		}
	}
	return texts
}

func hasImages(cs aimix.Contents) bool {
	for _, c := range cs {
		if _, ok := c.(aimix.ImageContent); ok {
			return true
		}
	}
	return false
}

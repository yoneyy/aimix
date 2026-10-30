package aimix

import (
	"encoding/json"
	"fmt"
)

// The concrete content and message types fill in an empty Type or Role when
// marshaled, so a hand-built value always decodes back to the same type.

func (c TextContent) MarshalJSON() ([]byte, error) {
	type plain TextContent
	c.Type = ContentTypeText
	return json.Marshal(plain(c))
}

func (c ThinkingContent) MarshalJSON() ([]byte, error) {
	type plain ThinkingContent
	c.Type = ContentTypeThinking
	return json.Marshal(plain(c))
}

func (c ImageContent) MarshalJSON() ([]byte, error) {
	type plain ImageContent
	c.Type = ContentTypeImage
	return json.Marshal(plain(c))
}

func (c ToolCallContent) MarshalJSON() ([]byte, error) {
	type plain ToolCallContent
	c.Type = ContentTypeToolCall
	if len(c.Arguments) == 0 {
		c.Arguments = json.RawMessage("{}")
	}
	return json.Marshal(plain(c))
}

func (m SystemMessage) MarshalJSON() ([]byte, error) {
	type plain SystemMessage
	m.Role = RoleSystem
	return json.Marshal(plain(m))
}

func (m UserMessage) MarshalJSON() ([]byte, error) {
	type plain UserMessage
	m.Role = RoleUser
	return json.Marshal(plain(m))
}

func (m AssistantMessage) MarshalJSON() ([]byte, error) {
	type plain AssistantMessage
	m.Role = RoleAssistant
	return json.Marshal(plain(m))
}

func (m ToolResultMessage) MarshalJSON() ([]byte, error) {
	type plain ToolResultMessage
	m.Role = RoleToolResult
	return json.Marshal(plain(m))
}

// UnmarshalJSON decodes each element into the concrete type named by its "type" field.
func (cs *Contents) UnmarshalJSON(data []byte) error {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return err
	}
	if raws == nil {
		*cs = nil
		return nil
	}
	out := make(Contents, 0, len(raws))
	for i, raw := range raws {
		var head struct {
			Type ContentType `json:"type"`
		}
		if err := json.Unmarshal(raw, &head); err != nil {
			return fmt.Errorf("aimix: content %d: %w", i, err)
		}
		var (
			c   Content
			err error
		)
		switch head.Type {
		case ContentTypeText:
			c, err = decodeAs[TextContent](raw)
		case ContentTypeThinking:
			c, err = decodeAs[ThinkingContent](raw)
		case ContentTypeImage:
			c, err = decodeAs[ImageContent](raw)
		case ContentTypeToolCall:
			c, err = decodeAs[ToolCallContent](raw)
		default:
			err = fmt.Errorf("unknown content type %q", head.Type)
		}
		if err != nil {
			return fmt.Errorf("aimix: content %d: %w", i, err)
		}
		out = append(out, c)
	}
	*cs = out
	return nil
}

// UnmarshalJSON decodes each element into the concrete type named by its "role" field.
func (ms *Messages) UnmarshalJSON(data []byte) error {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return err
	}
	if raws == nil {
		*ms = nil
		return nil
	}
	out := make(Messages, 0, len(raws))
	for i, raw := range raws {
		var head struct {
			Role Role `json:"role"`
		}
		if err := json.Unmarshal(raw, &head); err != nil {
			return fmt.Errorf("aimix: message %d: %w", i, err)
		}
		var (
			m   Message
			err error
		)
		switch head.Role {
		case RoleSystem:
			m, err = decodeAs[SystemMessage](raw)
		case RoleUser:
			m, err = decodeAs[UserMessage](raw)
		case RoleAssistant:
			m, err = decodeAs[AssistantMessage](raw)
		case RoleToolResult:
			m, err = decodeAs[ToolResultMessage](raw)
		default:
			err = fmt.Errorf("unknown message role %q", head.Role)
		}
		if err != nil {
			return fmt.Errorf("aimix: message %d: %w", i, err)
		}
		out = append(out, m)
	}
	*ms = out
	return nil
}

func decodeAs[T any](raw json.RawMessage) (T, error) {
	var v T
	err := json.Unmarshal(raw, &v)
	return v, err
}

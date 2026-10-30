package infra

import (
	"encoding/json"
	"fmt"
	"strings"
)

// RepairJSON fixes malformed string literals that models commonly produce: raw
// control characters inside strings are escaped, and a backslash before an
// invalid escape character is doubled.
func RepairJSON(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inString := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !inString {
			b.WriteByte(c)
			if c == '"' {
				inString = true
			}
			continue
		}
		switch {
		case c == '"':
			b.WriteByte(c)
			inString = false
		case c == '\\':
			if i+1 >= len(s) {
				b.WriteString(`\\`)
				continue
			}
			next := s[i+1]
			if next == 'u' && i+6 <= len(s) && isHex4(s[i+2:i+6]) {
				b.WriteString(s[i : i+6])
				i += 5
				continue
			}
			if strings.IndexByte(`"\/bfnrt`, next) >= 0 {
				b.WriteByte('\\')
				b.WriteByte(next)
				i++
				continue
			}
			b.WriteString(`\\`)
		case c < 0x20:
			b.WriteString(escapeControl(c))
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

// ParseJSONObject parses the complete text of a JSON object, retrying once on the
// RepairJSON output. Empty text is an empty object.
func ParseJSONObject(s string) (json.RawMessage, error) {
	if strings.TrimSpace(s) == "" {
		return json.RawMessage("{}"), nil
	}
	raw, err := parseObject(s)
	if err != nil {
		if repaired := RepairJSON(s); repaired != s {
			if raw, rerr := parseObject(repaired); rerr == nil {
				return raw, nil
			}
		}
		return nil, err
	}
	return raw, nil
}

func parseObject(s string) (json.RawMessage, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(s), &obj); err != nil {
		return nil, err
	}
	if obj == nil {
		return nil, fmt.Errorf("JSON value is null, want an object")
	}
	return json.RawMessage(strings.TrimSpace(s)), nil
}

func isHex4(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F') {
			return false
		}
	}
	return len(s) == 4
}

func escapeControl(c byte) string {
	switch c {
	case '\b':
		return `\b`
	case '\f':
		return `\f`
	case '\n':
		return `\n`
	case '\r':
		return `\r`
	case '\t':
		return `\t`
	}
	return fmt.Sprintf(`\u%04x`, c)
}

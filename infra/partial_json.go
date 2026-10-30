package infra

import (
	"encoding/json"
	"slices"
	"strings"
)

// PartialJSON scans a JSON document that arrives in pieces (such as streamed tool
// call arguments) and renders the valid JSON value implied by the text so far.
// Each Write only scans the new text.
type PartialJSON struct {
	buf    []byte
	stack  []jsonFrame
	broken bool

	rootDone bool
	inString bool
	isKey    bool
	escape   bool
	inScalar bool

	scalarStart int

	// The last point where the document could be closed: the prefix length and
	// the closing brackets it needs.
	hasCheck   bool
	checkLen   int
	checkClose string
}

type jsonFrame struct {
	open  byte
	state uint8
}

const (
	stateValue uint8 = iota // expecting a value (or ']' right after '[')
	stateKey                // expecting an object key (or '}' right after '{')
	stateColon              // after a key, expecting ':'
	stateAfter              // after a value, expecting ',' or the closing bracket
)

// RepairPartialJSON renders the valid JSON value implied by an incomplete
// document. See PartialJSON.Value.
func RepairPartialJSON(s string) (json.RawMessage, bool) {
	var p PartialJSON
	p.Write(s)
	return p.Value()
}

// Write appends the next piece of the document.
func (p *PartialJSON) Write(s string) {
	start := len(p.buf)
	p.buf = append(p.buf, s...)
	for i := start; i < len(p.buf) && !p.broken; i++ {
		if !p.step(i) {
			p.broken = true
		}
	}
}

// Reset discards everything written so far.
func (p *PartialJSON) Reset() {
	*p = PartialJSON{buf: p.buf[:0]}
}

// Text returns everything written so far.
func (p *PartialJSON) Text() string {
	return string(p.buf)
}

// Value returns the text so far completed into valid JSON: an unterminated string
// value is closed, a trailing incomplete key, literal or separator is dropped, and
// open containers are closed. It reports false when the text is empty or malformed.
// The returned slice is freshly allocated.
func (p *PartialJSON) Value() (json.RawMessage, bool) {
	if p.broken {
		return nil, false
	}
	closers := p.closers()
	switch {
	case p.inString && !p.isKey:
		body := p.buf
		if p.escape {
			body = body[:len(body)-1]
		}
		// Trimming up to 5 bytes drops a partial \uXXXX escape.
		for trim := 0; trim <= 5 && trim <= len(body); trim++ {
			if v, ok := validJoin(body[:len(body)-trim], `"`+closers); ok {
				return v, true
			}
		}
	case !p.inString:
		if v, ok := validJoin(p.buf, closers); ok {
			return v, true
		}
	}
	if p.hasCheck {
		return validJoin(p.buf[:p.checkLen], p.checkClose)
	}
	return nil, false
}

func validJoin(prefix []byte, suffix string) (json.RawMessage, bool) {
	out := make([]byte, 0, len(prefix)+len(suffix))
	out = append(out, prefix...)
	out = append(out, suffix...)
	if !json.Valid(out) {
		return nil, false
	}
	return out, true
}

func (p *PartialJSON) step(i int) bool {
	c := p.buf[i]
	if p.inString {
		switch {
		case p.escape:
			p.escape = false
		case c == '\\':
			p.escape = true
		case c == '"':
			p.inString = false
			if p.isKey {
				p.top().state = stateColon
			} else {
				p.valueDone(i + 1)
			}
		}
		return true
	}
	if p.inScalar {
		if isScalarByte(c) {
			return true
		}
		p.inScalar = false
		if !json.Valid(p.buf[p.scalarStart:i]) {
			return false
		}
		p.valueDone(i)
	}

	switch c {
	case ' ', '\t', '\n', '\r':
		return true
	case '{', '[':
		if !p.expectingValue() {
			return false
		}
		state := stateValue
		if c == '{' {
			state = stateKey
		}
		p.stack = append(p.stack, jsonFrame{
			open:  c,
			state: state,
		})
		p.checkpoint(i + 1)
	case '}', ']':
		top := p.top()
		if top == nil || top.open != openerOf(c) || top.state == stateColon {
			return false
		}
		if top.open == '{' && top.state == stateValue {
			return false
		}
		p.stack = p.stack[:len(p.stack)-1]
		p.valueDone(i + 1)
	case '"':
		if top := p.top(); top != nil && top.open == '{' && top.state == stateKey {
			p.isKey = true
		} else if p.expectingValue() {
			p.isKey = false
		} else {
			return false
		}
		p.inString = true
	case ':':
		top := p.top()
		if top == nil || top.state != stateColon {
			return false
		}
		top.state = stateValue
	case ',':
		top := p.top()
		if top == nil || top.state != stateAfter {
			return false
		}
		if top.open == '{' {
			top.state = stateKey
		} else {
			top.state = stateValue
		}
	default:
		if !isScalarStart(c) || !p.expectingValue() {
			return false
		}
		p.inScalar = true
		p.scalarStart = i
	}
	return true
}

func (p *PartialJSON) top() *jsonFrame {
	if len(p.stack) == 0 {
		return nil
	}
	return &p.stack[len(p.stack)-1]
}

func (p *PartialJSON) expectingValue() bool {
	top := p.top()
	if top == nil {
		return !p.rootDone
	}
	return top.state == stateValue
}

func (p *PartialJSON) valueDone(end int) {
	if top := p.top(); top != nil {
		top.state = stateAfter
	} else {
		p.rootDone = true
	}
	p.checkpoint(end)
}

func (p *PartialJSON) checkpoint(end int) {
	p.hasCheck = true
	p.checkLen = end
	p.checkClose = p.closers()
}

func (p *PartialJSON) closers() string {
	if len(p.stack) == 0 {
		return ""
	}
	var b strings.Builder
	for _, v := range slices.Backward(p.stack) {
		if v.open == '{' {
			b.WriteByte('}')
		} else {
			b.WriteByte(']')
		}
	}
	return b.String()
}

func openerOf(c byte) byte {
	if c == '}' {
		return '{'
	}
	return '['
}

func isScalarStart(c byte) bool {
	return c == '-' || (c >= '0' && c <= '9') || c == 't' || c == 'f' || c == 'n'
}

func isScalarByte(c byte) bool {
	return isScalarStart(c) || c == '.' || c == '+' || c == 'e' || c == 'E' ||
		c == 'r' || c == 'u' || c == 'a' || c == 'l' || c == 's'
}

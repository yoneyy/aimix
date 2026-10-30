package infra

import (
	"encoding/json"
	"testing"
)

func TestRepairPartialJSON(t *testing.T) {
	tests := []struct {
		in   string
		want string
		ok   bool
	}{
		{in: ``, ok: false},
		{in: `{`, want: `{}`, ok: true},
		{in: `{"a"`, want: `{}`, ok: true},
		{in: `{"a":`, want: `{}`, ok: true},
		{in: `{"a": tru`, want: `{}`, ok: true},
		{in: `{"a": true`, want: `{"a": true}`, ok: true},
		{in: `{"a": 12`, want: `{"a": 12}`, ok: true},
		{in: `{"a": 1.`, want: `{}`, ok: true},
		{in: `{"a": "hel`, want: `{"a": "hel"}`, ok: true},
		{in: `{"a": "x\`, want: `{"a": "x"}`, ok: true},
		{in: `{"a": "x\u00`, want: `{"a": "x"}`, ok: true},
		{in: `{"a":1,"b`, want: `{"a":1}`, ok: true},
		{in: `{"a":1,`, want: `{"a":1}`, ok: true},
		{in: `[1, 2,`, want: `[1, 2]`, ok: true},
		{in: `{"a": [1, {"b": "x`, want: `{"a": [1, {"b": "x"}]}`, ok: true},
		{in: `{"a": {"b": 1}, "c": [`, want: `{"a": {"b": 1}, "c": []}`, ok: true},
		{in: `{"a": 1}`, want: `{"a": 1}`, ok: true},
		{in: `{"a": 1}}`, ok: false},
		{in: `{"a" 1`, ok: false},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			got, ok := RepairPartialJSON(tt.in)
			if ok != tt.ok {
				t.Fatalf("ok = %v, want %v (got %s)", ok, tt.ok, got)
			}
			if ok && string(got) != tt.want {
				t.Fatalf("got %s, want %s", got, tt.want)
			}
			if ok && !json.Valid(got) {
				t.Fatalf("got invalid JSON %s", got)
			}
		})
	}
}

func TestPartialJSONIncremental(t *testing.T) {
	full := `{"path": "a.txt", "content": "line1\nline2", "n": [1, 2.5, null], "ok": false}`
	var p PartialJSON
	for i := 0; i < len(full); i++ {
		p.Write(full[i : i+1])
		v, ok := p.Value()
		if !ok {
			t.Fatalf("prefix %q: no value", full[:i+1])
		}
		if !json.Valid(v) {
			t.Fatalf("prefix %q: invalid JSON %s", full[:i+1], v)
		}
	}
	v, _ := p.Value()
	if string(v) != full {
		t.Fatalf("final value %s, want %s", v, full)
	}
	if p.Text() != full {
		t.Fatalf("Text() = %q", p.Text())
	}

	p.Reset()
	p.Write(`{"x": 1`)
	if v, _ := p.Value(); string(v) != `{"x": 1}` {
		t.Fatalf("after Reset got %s", v)
	}
}

func TestParseJSONObject(t *testing.T) {
	tests := []struct {
		in      string
		want    string
		wantErr bool
	}{
		{in: "", want: `{}`},
		{in: `  {"a": 1} `, want: `{"a": 1}`},
		{in: "{\"a\": \"line1\nline2\"}", want: `{"a": "line1\nline2"}`},
		{in: `{"p": "C:\dir"}`, want: `{"p": "C:\\dir"}`},
		{in: `[1]`, wantErr: true},
		{in: `null`, wantErr: true},
		{in: `{"a":`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			got, err := ParseJSONObject(tt.in)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && string(got) != tt.want {
				t.Fatalf("got %s, want %s", got, tt.want)
			}
		})
	}
}

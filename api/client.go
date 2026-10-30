package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
	openaissestream "github.com/openai/openai-go/v3/packages/ssestream"

	"github.com/yoneyy/aimix"
)

const defaultOpenAIBaseURL = "https://api.openai.com/v1"

var errMissingAPIKey = errors.New("aimix: missing API key")

// newAnthropicClient builds a client for one request. The SDK's environment
// defaults (ANTHROPIC_BASE_URL, ANTHROPIC_CUSTOM_HEADERS, ...) are disabled so
// that only the model and options decide where and how requests go. Model and
// option headers are applied after extra, so callers can override them.
func newAnthropicClient(model aimix.Model, opts aimix.StreamOptions, extra ...anthropicoption.RequestOption) anthropic.Client {
	ro := []anthropicoption.RequestOption{
		anthropicoption.WithoutEnvironmentDefaults(),
		anthropicoption.WithAPIKey(opts.APIKey),
		anthropicoption.WithMaxRetries(max(opts.MaxRetries, 0)),
	}
	if model.BaseURL != "" {
		ro = append(ro, anthropicoption.WithBaseURL(anthropicBaseURL(model.BaseURL)))
	}
	if opts.HTTPClient != nil {
		ro = append(ro, anthropicoption.WithHTTPClient(opts.HTTPClient))
	}
	ro = append(ro, extra...)
	for _, headers := range []map[string]string{model.Headers, opts.Headers} {
		for k, v := range headers {
			ro = append(ro, anthropicoption.WithHeader(k, v))
		}
	}
	return anthropic.NewClient(ro...)
}

// anthropicBaseURL drops a trailing "/v1": the SDK adds "/v1/messages" itself,
// but proxies shared with OpenAI-style clients are usually configured with it.
func anthropicBaseURL(u string) string {
	return strings.TrimSuffix(strings.TrimRight(u, "/"), "/v1")
}

// openAIRequestOptions configures OpenAI services for one request. The services
// are built directly instead of through openai.NewClient, which reads
// OPENAI_BASE_URL, OPENAI_ORG_ID and other environment variables.
func openAIRequestOptions(model aimix.Model, opts aimix.StreamOptions) []openaioption.RequestOption {
	baseURL := model.BaseURL
	if baseURL == "" {
		baseURL = defaultOpenAIBaseURL
	}
	var httpClient openaioption.HTTPClient = http.DefaultClient
	if opts.HTTPClient != nil {
		httpClient = opts.HTTPClient
	}
	ro := []openaioption.RequestOption{
		openaioption.WithBaseURL(baseURL),
		openaioption.WithAPIKey(opts.APIKey),
		openaioption.WithHTTPClient(httpClient),
		openaioption.WithMaxRetries(max(opts.MaxRetries, 0)),
	}
	for _, headers := range []map[string]string{model.Headers, opts.Headers} {
		for k, v := range headers {
			ro = append(ro, openaioption.WithHeader(k, v))
		}
	}
	return ro
}

// checkEventStream rejects a successful response that is an HTML page: the
// base URL points at a website instead of the API (for OpenAI-compatible
// servers, often a missing "/v1"). Other content types are let through, since
// some compatible servers label event streams loosely.
func checkEventStream(resp *http.Response) error {
	if resp == nil {
		return nil
	}
	mediaType, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if mediaType != "text/html" {
		return nil
	}
	url := ""
	if resp.Request != nil {
		url = resp.Request.URL.String()
	}
	return fmt.Errorf("aimix: %s returned an HTML page instead of an event stream; check Model.BaseURL", url)
}

// formatError renders SDK errors with the provider's message, which the SDK
// error strings leave out or bury in a dump of the request.
func formatError(err error) string {
	if anthropicErr, ok := errors.AsType[*anthropic.Error](err); ok {
		status := fmt.Sprint(anthropicErr.StatusCode)
		// Errors sent inside a stream keep the HTTP status of the stream (200).
		if anthropicErr.StatusCode < 400 && anthropicErr.Type() != "" {
			status = string(anthropicErr.Type())
		}
		return fmt.Sprintf("Anthropic API error (%s): %s", status, anthropicErrorMessage(anthropicErr))
	}

	if openaiErr, ok := errors.AsType[*openai.Error](err); ok {
		msg := openaiErr.Message
		if msg == "" {
			msg = strings.TrimSpace(openaiErr.RawJSON())
		}
		if msg == "" {
			msg = http.StatusText(openaiErr.StatusCode)
		}
		return fmt.Sprintf("OpenAI API error (%d): %s", openaiErr.StatusCode, msg)
	}

	// An {"error": ...} object sent inside the stream.
	if streamErr, ok := errors.AsType[*openaissestream.StreamError](err); ok {
		if msg := errorBodyMessage(string(streamErr.Event.Data)); msg != "" {
			return "OpenAI API error: " + msg
		}
		return "OpenAI API error: " + streamErr.Message
	}
	return err.Error()
}

func anthropicErrorMessage(err *anthropic.Error) string {
	raw := err.RawJSON()
	if msg := errorBodyMessage(raw); msg != "" {
		return msg
	}
	if raw = strings.TrimSpace(raw); raw != "" {
		return raw
	}
	return http.StatusText(err.StatusCode)
}

// errorBodyMessage extracts error.message from an error body.
func errorBodyMessage(raw string) string {
	var body struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(raw), &body) != nil {
		return ""
	}
	return body.Error.Message
}

package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("warning: .env file not found, using system environment")
	}

	http.HandleFunc("/recognize", recognizeHandler)

	addr := ":8080"
	log.Printf("Listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

func recognizeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	//=========PARSING_FORM=========//

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20)

	const maxMemory = 20 << 20
	if err := r.ParseMultipartForm(maxMemory); err != nil {
		http.Error(w, "failed parse multipart form: "+err.Error(), http.StatusBadRequest)
		return
	}

	file, fh, err := r.FormFile("photo")
	if err != nil {
		http.Error(w, "missing form file 'photo': "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	imgBytes, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "failed read file: "+err.Error(), http.StatusInternalServerError)
		return
	}

	mimeType := http.DetectContentType(imgBytes)
	if mimeType == "application/octet-stream" {
		if t := fh.Header.Get("Content-Type"); t != "" {
			mimeType = t
		}
	}

	//=========PROMPT=========//

	b64 := base64.StdEncoding.EncodeToString(imgBytes)
	dataURL := fmt.Sprintf("data:%s;base64,%s", mimeType, b64)

	promptFile := "prompts/car_identification.txt"
	promptText, err := loadPrompt(promptFile)
	if err != nil {
		log.Printf("failed to load prompt from file: %v", err)
		http.Error(w, "server error: failed to load prompt", http.StatusInternalServerError)
		return
	}

	bodyObj := map[string]interface{}{
		"model": "gpt-5-mini",
		"input": []interface{}{
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "input_text", "text": promptText},
					map[string]interface{}{"type": "input_image", "image_url": dataURL},
				},
			},
		},
	}

	jb, err := json.Marshal(bodyObj)
	if err != nil {
		http.Error(w, "failed marshal request body: "+err.Error(), http.StatusInternalServerError)
		return
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		http.Error(w, "server not configured: set OPENAI_API_KEY env var", http.StatusInternalServerError)
		return
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/responses", bytes.NewReader(jb))
	if err != nil {
		http.Error(w, "failed create request: "+err.Error(), http.StatusInternalServerError)
		return
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "request to OpenAI failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "failed read response: "+err.Error(), http.StatusInternalServerError)
		return
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		w.Write(respBytes)
		return
	}

	//=========PARSING_RESPONSE=========//

	cleanJSON, err := extractCarJSON(respBytes)
	if err != nil {
		log.Printf("extractCarJSON error: %v; full response: %s", err, string(respBytes))
		http.Error(w, "failed parse model output: "+err.Error(), http.StatusInternalServerError)
		return
	}

	//=========PHOTO=========//

	samplePhotoPath := "car2.jpg"

	photoBytes, err := os.ReadFile(samplePhotoPath)
	if err != nil {
		log.Printf("failed to read sample photo: %v", err)
		http.Error(w, "server error: cannot read sample photo", http.StatusInternalServerError)
		return
	}

	//=========MULTIPART=========//

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	photoPartHeader := textproto.MIMEHeader{}
	photoPartHeader.Set("Content-Disposition", `form-data; name="photo"; filename="sample_photo.jpg"`)
	photoPartHeader.Set("Content-Type", "image/jpeg")
	photoPart, _ := writer.CreatePart(photoPartHeader)
	photoPart.Write(photoBytes)

	jsonPartHeader := textproto.MIMEHeader{}
	jsonPartHeader.Set("Content-Disposition", `form-data; name="car_info"`)
	jsonPartHeader.Set("Content-Type", "application/json")
	jsonPart, _ := writer.CreatePart(jsonPartHeader)
	jsonPart.Write(cleanJSON)

	writer.Close()

	w.Header().Set("Content-Type", writer.FormDataContentType())
	w.WriteHeader(http.StatusOK)
	w.Write(buf.Bytes())
}

func extractJSONFromText(s string) (string, error) {
	s = strings.TrimSpace(s)

	var tmp interface{}
	if json.Unmarshal([]byte(s), &tmp) == nil {
		return s, nil
	}

	reFence := regexp.MustCompile("(?s)```(?:json\\s*)?(.*?)```")
	if m := reFence.FindStringSubmatch(s); len(m) >= 2 {
		candidate := strings.TrimSpace(m[1])
		if json.Unmarshal([]byte(candidate), &tmp) == nil {
			return candidate, nil
		}
	}

	start := strings.IndexAny(s, "{[")
	if start == -1 {
		return "", errors.New("no JSON object/array start found in text")
	}

	for end := len(s) - 1; end > start; end-- {
		if (s[end] == '}' && s[start] == '{') || (s[end] == ']' && s[start] == '[') {
			cand := strings.TrimSpace(s[start : end+1])
			if json.Unmarshal([]byte(cand), &tmp) == nil {
				return cand, nil
			}
		}
	}

	return "", errors.New("couldn't extract valid JSON substring from assistant text")
}

func extractCarJSON(respBytes []byte) ([]byte, error) {
	var apiResp struct {
		Output []struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}

	if err := json.Unmarshal(respBytes, &apiResp); err != nil {
		var generic map[string]interface{}
		if err2 := json.Unmarshal(respBytes, &generic); err2 != nil {
			return nil, fmt.Errorf("failed to parse OpenAI response: %w", err)
		}
		var found string
		var walk func(interface{})
		walk = func(v interface{}) {
			if found != "" {
				return
			}
			switch vv := v.(type) {
			case map[string]interface{}:
				for k, val := range vv {
					if k == "text" {
						if s, ok := val.(string); ok && strings.TrimSpace(s) != "" {
							found = s
							return
						}
					}
					walk(val)
					if found != "" {
						return
					}
				}
			case []interface{}:
				for _, item := range vv {
					walk(item)
					if found != "" {
						return
					}
				}
			}
		}
		walk(generic)
		if found == "" {
			return nil, errors.New("no output text found in response (generic parse)")
		}
		rawText := found
		jsonStr, err := extractJSONFromText(rawText)
		if err != nil {
			return nil, fmt.Errorf("failed to extract JSON from assistant text: %w", err)
		}
		var tmp interface{}
		if err := json.Unmarshal([]byte(jsonStr), &tmp); err != nil {
			return nil, fmt.Errorf("assistant text does not contain valid JSON: %w", err)
		}
		clean, _ := json.MarshalIndent(tmp, "", "  ")
		return clean, nil
	}

	var rawText string
	for _, out := range apiResp.Output {
		for _, c := range out.Content {
			if strings.TrimSpace(c.Text) != "" {
				rawText = c.Text
				break
			}
		}
		if rawText != "" {
			break
		}
	}

	if rawText == "" {
		return nil, errors.New("no output content with text found in response (maybe only reasoning entries present)")
	}

	jsonStr, err := extractJSONFromText(rawText)
	if err != nil {
		return nil, fmt.Errorf("failed to extract JSON from assistant text: %w", err)
	}

	var tmp interface{}
	if err := json.Unmarshal([]byte(jsonStr), &tmp); err != nil {
		return nil, fmt.Errorf("assistant text does not contain valid JSON: %w", err)
	}

	clean, err := json.MarshalIndent(tmp, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to reformat JSON: %w", err)
	}

	return clean, nil
}

func loadPrompt(filePath string) (string, error) {
	bytes, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

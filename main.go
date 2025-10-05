package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/joho/godotenv"
)

func main() {
	// Подгружаем .env
	if err := godotenv.Load(); err != nil {
		log.Println("warning: .env file not found, using system environment")
	}

	http.HandleFunc("/recognize", recognizeHandler)

	addr := ":8080"
	log.Printf("Listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

func recognizeHandler(w http.ResponseWriter, r *http.Request) {
	// Разрешаем только POST
	if r.Method != http.MethodPost {
		http.Error(w, "only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	// Ограничение общего тела запроса (пример: 10MB)
	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB

	// Парсим multipart (в памяти до maxMemory, остальное во временный файл)
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

	// Определим MIME (DetectContentType)
	mimeType := http.DetectContentType(imgBytes)
	if mimeType == "application/octet-stream" {
		if t := fh.Header.Get("Content-Type"); t != "" {
			mimeType = t
		}
	}

	// Кодируем в base64 и делаем data URL
	b64 := base64.StdEncoding.EncodeToString(imgBytes)
	dataURL := fmt.Sprintf("data:%s;base64,%s", mimeType, b64)

	// Формируем строгий промпт, требуем только JSON-ответ
	promptText := `You are an expert automobile identifier. Analyze the attached image and ONLY return a JSON object (no extra commentary) with the fields:
{
  "brand": string|null,
  "modelName": string|null,
  "generation": string|null,
  "years": string|null,
  "priceRange": string|null,
  "horsepower": number|null,
  "isElectric": boolean|null,
  "isHybrid": boolean|null
}
If unknown, use null.`

	// Тело для Responses API
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
		// "temperature": 0.0,
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

	// Возвращаем ответ OpenAI как есть (JSON). В продакшне лучше распарсить + валидация.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(respBytes)
}

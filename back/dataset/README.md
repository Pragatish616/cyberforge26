# Dataset Folder

Drop your training/reference data here. The backend supports three formats:

## 1. JSON files (`*.json`)

A JSON array of objects:

```json
[
  {
    "title": "STM32 UART Driver",
    "description": "USART2 at 115200 baud using HAL on PA2/PA3",
    "code": "#include \"stm32f4xx_hal.h\"\n...",
    "tags": ["uart", "stm32", "hal", "serial"]
  }
]
```

Required fields: `title`, `code`  
Optional fields: `description`, `tags`

## 2. C/C++ source files (`*.c`, `*.h`)

Drop any raw `.c` or `.h` files — the backend automatically splits them into
function-level chunks and indexes each chunk.

## 3. Multiple files

You can mix and match. All JSON files and all source files inside this folder
(including sub-folders) are loaded at startup.

To reload without restarting the server:
```
POST http://localhost:8000/reload
```

# Seraphim I — Local Multimodal AI Assistant

**Seraphim** è un assistente AI progettato per funzionare localmente tramite `llama.cpp`, con supporto alla visione artificiale attraverso webcam, ricerca sul web e interazione con un modello linguistico locale.

Il progetto si sviluppa attraverso diverse versioni, con l'obiettivo di migliorare velocità, efficienza e funzionalità.

## Funzionalità

- **AI locale:** comunicazione con un modello GGUF tramite `llama.cpp`.
- **Visione artificiale:** acquisizione e analisi delle immagini dalla webcam.
- **Ricerca web:** possibilità di effettuare ricerche online tramite DuckDuckGo.
- **Conversazione:** gestione del contesto della conversazione.
- **Ottimizzazione:** versioni alternative progettate per ridurre i tempi di risposta.
- **Text-to-Speech sperimentale:** integrazione del motore Kokoro per la sintesi vocale.

## Struttura del progetto

```text
Seraphim/
│
├── app.py
├── app_fast.py
├── app_fast_v2.py
│
├── prove/
│   ├── prova.py
│   ├── app_improved.py
│   └── aggiornamento.py
│
├── tts_kokoro.py
├── models/                  # Modelli locali, non inclusi in Git
├── README.md
├── .gitignore
└── LICENSE-LLVM-OpenMP
```

### Versioni disponibili

| File | Descrizione |
|---|---|
| `app.py` | Versione principale iniziale |
| `app_fast.py` | Versione con ottimizzazioni delle prestazioni |
| `app_fast_v2.py` | Ulteriore evoluzione della versione ottimizzata |
| `prove/app_improved.py` | Versione sperimentale con miglioramenti alla gestione delle richieste |
| `prove/aggiornamento.py` | Versione sperimentale con integrazione vocale |
| `prove/prova.py` | Test basilare della webcam |
| `tts_kokoro.py` | Modulo dedicato alla sintesi vocale Kokoro |

## Requisiti

- Windows (ambiente di sviluppo di riferimento).
- Python 3.10 o superiore.
- `llama.cpp` con supporto CUDA, se si desidera utilizzare la GPU NVIDIA.
- Webcam per le funzionalità di visione.
- Modello GGUF compatibile con la visione, insieme al relativo file `mmproj`.

## Installazione

### 1. Clonare il repository

```bash
git clone https://github.com/TabellaBrodos/Seraphim.git
cd Seraphim
```

### 2. Installare le dipendenze

Per le versioni principali:

```bash
python -m pip install opencv-python requests duckduckgo_search
```

Per la versione sperimentale con sintesi vocale:

```bash
python -m pip install numpy sounddevice kokoro
```

Alcune versioni supportano anche il pacchetto `ddgs` al posto di `duckduckgo_search`.

### 3. Configurare i modelli

Creare una cartella `models` nella directory principale del progetto e inserirvi:

- Il modello GGUF compatibile con la visione.
- Il relativo modello di proiezione multimodale (`mmproj`).

I modelli non sono inclusi nel repository a causa delle loro dimensioni.

## Avvio

### 1. Avviare il server llama.cpp

Esempio di comando:

```bash
llama-server -m models/Qwen3VL-4B-Instruct-Q4_K_M.gguf --mmproj models/mmproj-Qwen3VL-4B-Instruct-F16.gguf -ngl 99 --flash-attn on --port 8080 --ctx-size 4096
```

Il server espone l'API locale all'indirizzo:

```text
http://127.0.0.1:8080/v1/chat/completions
```

Assicurarsi che il server sia attivo prima di avviare Seraphim.

### 2. Avviare Seraphim

Dalla directory principale:

```bash
python app.py
```

Per provare le versioni ottimizzate:

```bash
python app_fast.py
```

oppure:

```bash
python app_fast_v2.py
```

Per provare la versione sperimentale con Kokoro:

```bash
python prove/aggiornamento.py
```

## Configurazione

Alcuni parametri, come indirizzo del server, risoluzione della webcam, qualità delle immagini e impostazioni vocali, sono definiti direttamente nei rispettivi file Python.

Le versioni possono avere configurazioni e funzionalità differenti.

## Stato dello sviluppo

Seraphim è un progetto in evoluzione.

Le versioni presenti nel repository rappresentano diverse fasi di sviluppo e sperimentazione.

**Funzionalità implementate o presenti nel codice:**
- Comunicazione con un LLM locale.
- Acquisizione delle immagini tramite webcam.
- Ricerca web.
- Ottimizzazioni delle richieste e delle immagini.
- Sperimentazione con sintesi vocale.

**Aree di miglioramento:**
- Consolidamento delle diverse versioni in un'unica applicazione.
- Gestione centralizzata della configurazione.
- Miglioramento della documentazione.
- Gestione delle dipendenze tramite `requirements.txt`.
- Test automatici e verifica della compatibilità tra sistemi.

## Note

- È necessario avviare `llama-server` prima dell'applicazione.
- I modelli AI devono essere scaricati e configurati separatamente.
- Le funzionalità possono variare a seconda della versione utilizzata.
- Le versioni presenti nella cartella `prove/` sono da considerarsi sperimentali.

## Licenza

Il repository include il file `LICENSE-LLVM-OpenMP`. Per le condizioni di licenza del progetto e delle singole dipendenze, fare riferimento ai rispettivi file e alle licenze dei componenti utilizzati.

## File e cartelle non inclusi nel repository

Per motivi di dimensioni e gestione dell'ambiente di sviluppo, alcune cartelle non sono incluse nel repository GitHub:

- **`models/`** — contiene i modelli AI in formato GGUF e i relativi file `mmproj`. Devono essere scaricati e inseriti manualmente.
- **`venv/`** — ambiente virtuale Python. Deve essere creato localmente e utilizzato per installare le dipendenze del progetto.

### Creazione dell'ambiente virtuale

Dalla directory principale del progetto, eseguire:

```bash
python -m venv venv
```

Attivazione su Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Installazione delle dipendenze:

```bash
python -m pip install opencv-python requests duckduckgo_search
```

Per la versione sperimentale con Kokoro, installare anche le dipendenze vocali:

```bash
python -m pip install numpy sounddevice kokoro
```

**Nota:** né i modelli AI né l'ambiente virtuale sono scaricati automaticamente clonando il repository. Devono essere preparati separatamente prima dell'avvio dell'applicazione.

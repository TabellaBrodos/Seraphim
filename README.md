# Seraphim I --- Assistente IA Multimodale Locale

Assistente IA multimodale locale in tempo reale, pensato per sistemi con
**NVIDIA RTX 4060**, basato su `llama.cpp` (`llama-server`) e Python.

## Caratteristiche

-   **Streaming in tempo reale:** ricezione progressiva delle risposte
    dal server locale.
-   **Smart Vision:** possibile analisi della webcam in base alle
    richieste dell'utente.
-   **Ricerca web:** integrazione con DuckDuckGo per recuperare
    informazioni aggiornate.
-   **Memoria conversazionale:** gestione dello storico dei messaggi
    durante la conversazione.

> Nota: le caratteristiche effettivamente disponibili dipendono dal
> contenuto di `app.py` e dalla configurazione del server.

## Requisiti

-   Python 3.10 o superiore
-   `llama.cpp` con supporto CUDA, se si utilizza una GPU NVIDIA
-   Un modello GGUF compatibile
-   Il relativo file `mmproj`, se richiesto dal modello multimodale

## Installazione

Apri PowerShell nella cartella del progetto e installa le dipendenze
Python:

``` powershell
python -m pip install --upgrade pip
pip install opencv-python requests duckduckgo_search
```

Se il progetto utilizza altre librerie, aggiungile al comando oppure
crea un file `requirements.txt`.

## Avvio

1.  Avvia `llama-server.exe` con il modello e il proiettore visivo
    corretti.
2.  Verifica che il server sia in ascolto sull'indirizzo e sulla porta
    configurati in `app.py`.
3.  Avvia l'applicazione:

``` powershell
python app.py
```

## Modelli e file di grandi dimensioni

I modelli GGUF e gli altri file pesanti non sono inclusi in questo
repository. Inseriscili localmente nella cartella `models/` e scaricali
separatamente seguendo le istruzioni del progetto.

## Sicurezza

Non caricare password, token API, chiavi private o file `.env`.
Controlla sempre `git status` prima del commit.


llama-server -m models/Qwen3VL-4B-Instruct-Q4_K_M.gguf --mmproj models/mmproj-Qwen3VL-4B-Instruct-F16.gguf -ngl 99 --flash-attn on --port 8080 --ctx-size 4096
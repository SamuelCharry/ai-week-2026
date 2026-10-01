# Pipeline Machine B. Cada paso lee/escribe en process/ y es reejecutable.
# Uso:  make all | make verify | make download | ...

PY := python3
SCRIPT_DIR := scripts

.PHONY: all verify download markdown chunk enrich index clean deps

all: verify download markdown chunk enrich index
	@echo
	@echo "========================================================"
	@echo "  Pipeline completo. Copia a Machine A:"
	@echo "    process/05_index.faiss"
	@echo "    process/05_metadata.sqlite"
	@echo "    process/manifest_v2.json"
	@echo "========================================================"

deps:
	$(PY) -m pip install -r requirements.txt

verify:
	$(PY) $(SCRIPT_DIR)/00_verify_gaps.py

download: verify
	$(PY) $(SCRIPT_DIR)/01_download.py

markdown: download
	$(PY) $(SCRIPT_DIR)/02_to_markdown.py

chunk: markdown
	$(PY) $(SCRIPT_DIR)/03_chunk.py

enrich: chunk
	$(PY) $(SCRIPT_DIR)/04_contextual_enrich.py

index: enrich
	$(PY) $(SCRIPT_DIR)/05_embed_index.py

clean:
	rm -rf process/01_downloaded process/02_markdown
	rm -f process/00_gaps.json process/00_gaps.md
	rm -f process/01_downloads.jsonl process/03_chunks.jsonl
	rm -f process/04_chunks_enriched.jsonl
	rm -f process/05_index.faiss process/05_metadata.sqlite process/manifest_v2.json

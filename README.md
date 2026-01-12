# Desafio MBA Engenharia de Software com IA - Full Cycle

Descreva abaixo como executar a sua solução.

## Subir infraestrutura (Postgres + pgvector + Adminer)
1) Inicie os contêineres:
```
docker compose up -d
```

2) Verifique saúde do Postgres:
```
docker compose ps
```
O serviço `postgres` deve estar `healthy`.

3) Acesse o Adminer (opcional) em `http://localhost:8080` (Server: `postgres`, user `postgres`, senha `postgres`, db `rag`).

4) Confirme a extensão `vector`:
```
docker compose exec postgres psql -U postgres -d rag -c "SELECT extname FROM pg_extension WHERE extname='vector';"
```

5) Pare/desligue:
```
docker compose down
```

## Ativar o ambiente Python (opcional para scripts)
```
source venv/bin/activate
```

## Ingestão de PDF para pgvector
Variáveis de ambiente necessárias:
- `OPENAI_API_KEY`: chave da OpenAI.
- `PDF_PATH`: caminho absoluto do PDF a ingerir.
- (opcional) `DATABASE_URL`: conexão Postgres. Padrão: `postgresql://postgres:postgres@localhost:5432/rag`.

Passos:
1) Garanta Postgres ativo (`docker compose up -d`).
2) Ative o venv (se ainda não): `source venv/bin/activate`
3) Rode a ingestão:
```
python src/ingest.py
```
O script divide o PDF em chunks de 1000 caracteres com overlap de 150, gera embeddings com `text-embedding-3-small` e grava no Postgres em `documents` (pgvector).
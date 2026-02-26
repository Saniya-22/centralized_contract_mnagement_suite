/**
 * Migration script to add PostgreSQL Full-Text Search (FTS) to embeddings_dense table.
 *
 * This replaces the custom BM25 sparse search with PostgreSQL's built-in tsvector/tsquery
 * which provides proper tokenization, stemming, stop words, IDF-based ranking, and
 * document length normalization.
 *
 * No re-ingestion needed - populates the FTS column from existing text data.
 *
 * Usage: yarn migrate:fts
 */

import pg from 'pg';
import dotenv from 'dotenv';

dotenv.config();

const {
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS, PG_SSLMODE, PG_DENSE_TABLE
} = process.env;

const ssl = PG_SSLMODE === 'require' ? { rejectUnauthorized: false } : false;
const tableName = PG_DENSE_TABLE || 'embeddings_dense';

async function migrate() {
    const pool = new pg.Pool({
        host: PG_HOST,
        port: parseInt(PG_PORT || '5432'),
        database: PG_DB,
        user: PG_USER,
        password: PG_PASS,
        ssl,
    });

    const client = await pool.connect();

    try {
        console.log('[FTS Migration] Starting...');
        console.log(`[FTS Migration] Target table: ${tableName}`);

        // 1. Add search_vector column if it doesn't exist
        console.log('[FTS Migration] Adding search_vector column...');
        await client.query(`
            ALTER TABLE ${tableName}
            ADD COLUMN IF NOT EXISTS search_vector tsvector;
        `);

        // 2. Populate search_vector from existing text column
        console.log('[FTS Migration] Populating search_vector from existing text...');
        const updateResult = await client.query(`
            UPDATE ${tableName}
            SET search_vector = to_tsvector('english', text)
            WHERE search_vector IS NULL;
        `);
        console.log(`[FTS Migration] Updated ${updateResult.rowCount} rows.`);

        // 3. Create GIN index for fast full-text search
        console.log('[FTS Migration] Creating GIN index on search_vector...');
        await client.query(`
            CREATE INDEX IF NOT EXISTS idx_${tableName}_search_vector
            ON ${tableName}
            USING GIN (search_vector);
        `);

        // 4. Create trigger for auto-population on future inserts/updates
        console.log('[FTS Migration] Creating trigger for auto-population...');
        await client.query(`
            CREATE OR REPLACE FUNCTION ${tableName}_search_vector_trigger()
            RETURNS trigger AS $$
            BEGIN
                NEW.search_vector := to_tsvector('english', COALESCE(NEW.text, ''));
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        `);

        await client.query(`
            DROP TRIGGER IF EXISTS trg_${tableName}_search_vector ON ${tableName};
        `);

        await client.query(`
            CREATE TRIGGER trg_${tableName}_search_vector
            BEFORE INSERT OR UPDATE OF text ON ${tableName}
            FOR EACH ROW
            EXECUTE FUNCTION ${tableName}_search_vector_trigger();
        `);

        // 5. Verification: test query
        console.log('\n[FTS Migration] Running verification query: "differing site conditions"...');
        const testResult = await client.query(`
            SELECT
                id,
                namespace,
                metadata->>'source' AS source,
                metadata->>'part' AS part,
                ts_rank_cd(search_vector, query, 32) AS rank,
                LEFT(text, 200) AS preview
            FROM ${tableName},
                 plainto_tsquery('english', 'differing site conditions') AS query
            WHERE search_vector @@ query
              AND namespace = 'public-regulations'
            ORDER BY ts_rank_cd(search_vector, query, 32) DESC
            LIMIT 5;
        `);

        if (testResult.rows.length > 0) {
            console.log(`[FTS Migration] Found ${testResult.rows.length} results:`);
            testResult.rows.forEach((row, i) => {
                console.log(`  ${i + 1}. [${row.source} Part ${row.part}] rank=${row.rank}`);
                console.log(`     ${row.preview}...`);
            });
        } else {
            console.log('[FTS Migration] No results found for test query (this may be expected if no matching documents are ingested yet).');
        }

        // 6. Stats
        const stats = await client.query(`
            SELECT
                COUNT(*) AS total_rows,
                COUNT(search_vector) AS fts_populated
            FROM ${tableName};
        `);
        console.log(`\n[FTS Migration] Stats: ${stats.rows[0].fts_populated}/${stats.rows[0].total_rows} rows have FTS vectors.`);

        console.log('[FTS Migration] Done!');
    } catch (error) {
        console.error('[FTS Migration] Error:', error.message);
        throw error;
    } finally {
        client.release();
        await pool.end();
    }
}

migrate().catch(err => {
    console.error(err);
    process.exit(1);
});

#!/usr/bin/env node

import pg from 'pg';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const {
    PG_HOST = 'localhost',
    PG_PORT = '5432',
    PG_DB = 'daedalus',
    PG_USER = 'daedalus_admin',
    PG_PASS = '',
    PG_SSLMODE = 'disable'
} = process.env;

async function initializeDatabase() {
    console.log('🔧 Database Initialization Script');
    console.log('================================');
    console.log(`Host: ${PG_HOST}`);
    console.log(`Port: ${PG_PORT}`);
    console.log(`Database: ${PG_DB}`);
    console.log(`User: ${PG_USER}`);
    console.log(`SSL Mode: ${PG_SSLMODE}`);
    console.log('');

    const pool = new pg.Pool({
        host: PG_HOST,
        port: parseInt(PG_PORT),
        database: PG_DB,
        user: PG_USER,
        password: PG_PASS,
        ssl: PG_SSLMODE === 'require' ? { rejectUnauthorized: false } : false
    });

    try {
        console.log('Testing database connection...');
        const client = await pool.connect();
        const result = await client.query('SELECT version()');
        console.log(`✅ Connected to: ${result.rows[0].version.split(',')[0]}`);
        console.log('');

        console.log('Checking pgvector extension...');
        const extResult = await client.query(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        );

        if (extResult.rows.length === 0) {
            console.log('Creating pgvector extension...');
            await client.query('CREATE EXTENSION IF NOT EXISTS vector');
            console.log('✅ pgvector extension created');
        } else {
            console.log('✅ pgvector extension already exists');
        }
        console.log('');

        console.log('Checking tables...');
        const tableCheck = await client.query(`
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('embeddings_dense', 'embeddings_sparse')
        `);

        const existingTables = tableCheck.rows.map(r => r.table_name);
        console.log(`Existing tables: ${existingTables.join(', ') || 'none'}`);

        if (!existingTables.includes('embeddings_dense') || !existingTables.includes('embeddings_sparse')) {
            console.log('');
            console.log('Creating tables...');

            const setupSqlPath = path.join(__dirname, 'setupRegulationsDB.sql');
            if (fs.existsSync(setupSqlPath)) {
                const setupSql = fs.readFileSync(setupSqlPath, 'utf8');
                await client.query(setupSql);
                console.log('✅ Tables created from setupRegulationsDB.sql');
            } else {
                await client.query(`
                    CREATE TABLE IF NOT EXISTS embeddings_dense (
                        id SERIAL PRIMARY KEY,
                        namespace VARCHAR(255) NOT NULL,
                        chunk_id VARCHAR(255) NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector(1536),
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS embeddings_sparse (
                        id SERIAL PRIMARY KEY,
                        namespace VARCHAR(255) NOT NULL,
                        chunk_id VARCHAR(255) NOT NULL,
                        content TEXT NOT NULL,
                        bm25_indices JSONB NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_dense_namespace ON embeddings_dense(namespace);
                    CREATE INDEX IF NOT EXISTS idx_sparse_namespace ON embeddings_sparse(namespace);
                    CREATE INDEX IF NOT EXISTS idx_dense_metadata ON embeddings_dense USING gin(metadata);
                    CREATE INDEX IF NOT EXISTS idx_sparse_metadata ON embeddings_sparse USING gin(metadata);
                `);
                console.log('✅ Tables created with inline SQL');
            }
        } else {
            console.log('✅ Tables already exist');
        }
        console.log('');

        console.log('Checking data...');
        const denseCount = await client.query('SELECT COUNT(*) FROM embeddings_dense');
        const sparseCount = await client.query('SELECT COUNT(*) FROM embeddings_sparse');
        console.log(`Dense embeddings: ${denseCount.rows[0].count}`);
        console.log(`Sparse embeddings: ${sparseCount.rows[0].count}`);

        client.release();

        console.log('');
        console.log('================================');
        console.log('✅ Database initialization complete!');

        return {
            success: true,
            denseCount: parseInt(denseCount.rows[0].count),
            sparseCount: parseInt(sparseCount.rows[0].count)
        };
    } catch (error) {
        console.error('❌ Database initialization failed:', error.message);
        throw error;
    } finally {
        await pool.end();
    }
}

const isMainModule = import.meta.url === `file://${process.argv[1]}`;
if (isMainModule) {
    initializeDatabase()
        .then(result => {
            if (result.denseCount === 0) {
                console.log('');
                console.log('⚠️  No data found. Run ingestion script to populate:');
                console.log('   node src/scripts/ingestRegulations.js');
            }
            process.exit(0);
        })
        .catch(error => {
            console.error('❌ initDB FATAL ERROR:', error.message);
            console.error(error.stack);
            process.exit(1);
        });
}

export { initializeDatabase };

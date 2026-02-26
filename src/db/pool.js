import pg from 'pg';

const {
    PG_HOST = 'localhost',
    PG_PORT = '5432',
    PG_DB = 'daedalus',
    PG_USER,
    PG_PASS,
    PG_SSLMODE = 'disable'
} = process.env;

const ssl = PG_SSLMODE === 'require' ? { rejectUnauthorized: false } : false;

export const pool = new pg.Pool({
    host: PG_HOST,
    port: parseInt(PG_PORT, 10),
    database: PG_DB,
    user: PG_USER,
    password: PG_PASS,
    ssl
});

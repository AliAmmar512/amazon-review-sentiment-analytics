CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE products (
    product_id VARCHAR(20) PRIMARY KEY,  -- using ASIN as the ID
    category_id INTEGER REFERENCES categories(category_id),
    product_title TEXT
);

CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    product_id VARCHAR(20) REFERENCES products(product_id),
    rating NUMERIC(2,1),
    review_text TEXT,
    sentiment VARCHAR(10),
    helpful_vote INTEGER DEFAULT 0,
    verified_purchase BOOLEAN,
    review_timestamp BIGINT
);

CREATE INDEX idx_reviews_product_id ON reviews(product_id);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment);
CREATE INDEX idx_reviews_category ON products(category_id);
-- @Author: [您的姓名/公司名称]
-- @Date: 2025-05-27
-- @Description: 针对大规模数据的 Swing 召回算法的 Hive SQL 实现。
-- 假设输入数据位于名为 'user_interactions' 的表中，包含以下列：
-- user_id STRING, item_id STRING, timestamp BIGINT (或 STRING，根据实际情况调整)
-- 步骤 1: 计算每个物品的独立用户数 (|U(i)|)
-- 此表将用于 Swing 公式中获取 |U(i)| 和 |U(j)|。
CREATE TEMPORARY TABLE IF NOT EXISTS item_user_counts AS
SELECT
    item_id,
    COUNT(DISTINCT user_id) AS user_count
FROM
    user_interactions
GROUP BY
    item_id;

-- 步骤 2: 生成同一用户交互过的所有唯一物品对
-- 这一步对于处理大型数据集的性能至关重要。
-- 我们按用户分组并收集所有物品，然后通过 EXPLODE 生成物品对。
CREATE TEMPORARY TABLE IF NOT EXISTS user_item_pairs AS
SELECT
    t1.item_id AS item1,
    t2.item_id AS item2,
    t.user_id
FROM
    (
        SELECT
            user_id,
            COLLECT_LIST(item_id) AS items
        FROM
            user_interactions
        GROUP BY
            user_id
    ) t LATERAL VIEW EXPLODE(t.items) a AS item_a LATERAL VIEW EXPLODE(t.items) b AS item_b
WHERE
    item_a < item_b
    AND item_a != item_b;

-- 确保 item1 < item2 以避免重复对和自配对
-- 步骤 3: 计算每个物品对的共同出现用户数 (|U(i) & U(j)|)
CREATE TEMPORARY TABLE IF NOT EXISTS co_occurrence_counts AS
SELECT
    item1,
    item2,
    COUNT(DISTINCT user_id) AS co_occurrence_count
FROM
    user_item_pairs
GROUP BY
    item1,
    item2;

-- 步骤 4: 计算每个物品对的 Swing 分数
-- Swing(i, j) = |U(i) & U(j)| / (|U(i)| + |U(j)| - |U(i) & U(j)|)
CREATE TABLE IF NOT EXISTS swing_scores AS
SELECT
    t1.item1,
    t1.item2,
    t1.co_occurrence_count,
    t2.user_count AS item1_user_count,
    t3.user_count AS item2_user_count,
    CAST(t1.co_occurrence_count AS DOUBLE) / (
        t2.user_count + t3.user_count - t1.co_occurrence_count
    ) AS swing_score
FROM
    co_occurrence_counts t1
    JOIN item_user_counts t2 ON t1.item1 = t2.item_id
    JOIN item_user_counts t3 ON t1.item2 = t3.item_id
WHERE
    (
        t2.user_count + t3.user_count - t1.co_occurrence_count
    ) > 0;

-- 避免除以零
-- 步骤 5: 为每个物品生成 Top-K 推荐
-- 这一步使用窗口函数根据 Swing 分数对物品进行排名。
CREATE TABLE IF NOT EXISTS swing_recommendations AS
SELECT
    item1,
    item2,
    swing_score,
    ROW_NUMBER() OVER (
        PARTITION BY item1
        ORDER BY
            swing_score DESC
    ) AS rank
FROM
    swing_scores QUALIFY ROW_NUMBER() OVER (
        PARTITION BY item1
        ORDER BY
            swing_score DESC
    ) <= 10;

-- Top 10 推荐，可根据需要调整
-- 清理临时表（可选，但推荐）
-- DROP TABLE IF EXISTS item_user_counts;
-- DROP TABLE IF EXISTS user_item_pairs;
-- DROP TABLE IF EXISTS co_occurrence_counts;
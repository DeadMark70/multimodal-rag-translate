-- Mark summaries at commit, after source-row updates. This avoids taking a
-- campaign revision lock before another writer's accounting/source row locks.
DO $$
DECLARE table_name text;
BEGIN
    FOR table_name IN
        SELECT event_object_table FROM information_schema.triggers
        WHERE trigger_schema=current_schema() AND trigger_name='analysis_changed'
        GROUP BY event_object_table
    LOOP
        EXECUTE format('DROP TRIGGER analysis_changed ON %I', table_name);
        EXECUTE format(
            'CREATE CONSTRAINT TRIGGER analysis_changed AFTER INSERT OR UPDATE OR DELETE ON %I DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION evaluation_analysis_changed()',
            table_name
        );
    END LOOP;
END $$;

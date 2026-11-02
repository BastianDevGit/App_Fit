-- Row Level Security para PostgreSQL (A7, HU-092).
-- La API fija el tenant y el usuario de la petición con:
--   SET LOCAL app.tenant_id = '<tenant>'; SET LOCAL app.user_id = '<uuid>';
-- En el MVP solo existe el rol "usuario"; coach y admin de gimnasio quedan modelados para el Modo Coach.

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['e1rm_history','mesocycles','performed_set_events','session_logs','daily_readiness','food_logs']
  LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format($p$CREATE POLICY tenant_isolation ON %I
      USING (tenant_id = current_setting('app.tenant_id', true)
             AND user_id = current_setting('app.user_id', true))$p$, t);
  END LOOP;
END $$;

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE users FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON users
  USING (tenant_id = current_setting('app.tenant_id', true) AND id = current_setting('app.user_id', true));

-- Series ejecutadas append-only (A1): la aplicación no tiene permiso de UPDATE.
REVOKE UPDATE ON performed_set_events FROM PUBLIC;

import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Stack,
  Alert,
  FormControlLabel,
  Switch,
} from "@mui/material";
import { useState } from "react";
import { useAppStore } from "../../stores/appStore";

export function Settings() {
  const { apiBaseUrl, authSecret, theme, setApiBaseUrl, setAuthSecret, setTheme } =
    useAppStore();
  const [localApiUrl, setLocalApiUrl] = useState(apiBaseUrl);
  const [localAuthSecret, setLocalAuthSecret] = useState(authSecret || "");
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setApiBaseUrl(localApiUrl);
    if (localAuthSecret.trim()) {
      setAuthSecret(localAuthSecret);
    } else {
      setAuthSecret(null);
    }
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <Box>
      <Paper elevation={1} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          设置
        </Typography>

        <Stack spacing={3} sx={{ mt: 3 }}>
          <TextField
            fullWidth
            label="API 服务地址"
            value={localApiUrl}
            onChange={(e) => setLocalApiUrl(e.target.value)}
            helperText="后端 FastAPI 服务的地址"
          />

          <TextField
            fullWidth
            label="认证密钥 (可选)"
            type="password"
            value={localAuthSecret}
            onChange={(e) => setLocalAuthSecret(e.target.value)}
            helperText="如果后端启用了认证，请输入 AUTH_SECRET"
          />

          <FormControlLabel
            control={
              <Switch
                checked={theme === "dark"}
                onChange={(e) => setTheme(e.target.checked ? "dark" : "light")}
              />
            }
            label="深色主题"
          />

          <Button variant="contained" onClick={handleSave}>
            保存设置
          </Button>

          {saved && (
            <Alert severity="success">设置已保存</Alert>
          )}
        </Stack>
      </Paper>
    </Box>
  );
}


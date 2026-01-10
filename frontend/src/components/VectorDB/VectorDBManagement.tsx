import {
  Box,
  Paper,
  Typography,
  Button,
  Stack,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  LinearProgress,
} from "@mui/material";
import { useState, useEffect } from "react";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { useVectorDBStore } from "../../stores/vectorDBStore";
import { apiClient } from "../../api/client";
import { useChatStore } from "../../stores/chatStore";

export function VectorDBManagement() {
  const { databases, currentDatabase } = useVectorDBStore();
  const { setCurrentAgent } = useChatStore();
  const [error, setError] = useState<string | null>(null);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dbName, setDbName] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  // 设置默认使用 rag-assistant agent
  useEffect(() => {
    apiClient
      .getInfo()
      .then((info) => {
        const ragAgent = info.agents.find((a) => a.key.includes("rag"));
        if (ragAgent) {
          setCurrentAgent(ragAgent.key);
        }
      })
      .catch((err) => {
        setError(`无法连接到服务: ${err.message}`);
      });
  }, [setCurrentAgent]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setSelectedFiles(Array.from(event.target.files));
    }
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0 || !dbName.trim()) {
      setError("请选择文件并输入数据库名称");
      return;
    }

    setUploading(true);
    setError(null);

    try {
      // 这里需要通过 agent 调用来上传文件
      // 由于后端 API 可能需要通过 agent 调用，我们使用聊天方式
      setError("文件上传功能需要通过 Agent 调用实现。请在聊天界面使用 rag-assistant agent 上传文件。");
      setUploadDialogOpen(false);
      setSelectedFiles([]);
      setDbName("");
    } catch (err) {
      setError(`上传失败: ${err instanceof Error ? err.message : "未知错误"}`);
    } finally {
      setUploading(false);
    }
  };

  const handleSwitchDB = async (_dbPath: string) => {
    try {
      // 通过 agent 调用切换数据库
      setError("数据库切换功能需要通过 Agent 调用实现。请在聊天界面使用 rag-assistant agent 切换数据库。");
    } catch (err) {
      setError(`切换失败: ${err instanceof Error ? err.message : "未知错误"}`);
    }
  };

  return (
    <Box>
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
          <Typography variant="h6">向量数据库管理</Typography>
          <Button
            variant="contained"
            startIcon={<CloudUploadIcon />}
            onClick={() => setUploadDialogOpen(true)}
          >
            创建数据库
          </Button>
        </Stack>
        <Typography variant="body2" color="text.secondary">
          管理向量数据库，上传文件创建新的数据库，或切换当前使用的数据库。
        </Typography>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Paper elevation={1} sx={{ p: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          数据库列表
        </Typography>
        {databases.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            暂无数据库。请通过聊天界面使用 rag-assistant agent 创建数据库。
          </Typography>
        ) : (
          <List>
            {databases.map((db, index) => (
              <ListItem key={index}>
                <ListItemText
                  primary={db.name}
                  secondary={`路径: ${db.path} | 类型: ${db.type}`}
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    onClick={() => handleSwitchDB(db.path)}
                    color={currentDatabase?.path === db.path ? "primary" : "default"}
                  >
                    {currentDatabase?.path === db.path ? (
                      <CheckCircleIcon />
                    ) : (
                      <RadioButtonUncheckedIcon />
                    )}
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        )}
      </Paper>

      <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>创建向量数据库</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="数据库名称"
            value={dbName}
            onChange={(e) => setDbName(e.target.value)}
            margin="normal"
            required
          />
          <Button variant="outlined" component="label" fullWidth sx={{ mt: 2 }}>
            选择文件
            <input type="file" hidden multiple onChange={handleFileSelect} />
          </Button>
          {selectedFiles.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2">已选择 {selectedFiles.length} 个文件:</Typography>
              {selectedFiles.map((file, index) => (
                <Typography key={index} variant="caption" display="block">
                  {file.name}
                </Typography>
              ))}
            </Box>
          )}
          {uploading && <LinearProgress sx={{ mt: 2 }} />}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)}>取消</Button>
          <Button onClick={handleUpload} variant="contained" disabled={uploading}>
            上传
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}


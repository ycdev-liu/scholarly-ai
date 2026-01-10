import { FormControl, InputLabel, Select, MenuItem, type SelectChangeEvent } from "@mui/material";
import type { ModelName } from "../../api/types";

interface ModelSelectorProps {
  models: ModelName[];
  selectedModel: ModelName | null;
  onModelChange: (model: ModelName | null) => void;
}

export function ModelSelector({ models, selectedModel, onModelChange }: ModelSelectorProps) {
  const handleChange = (event: SelectChangeEvent<string>) => {
    const value = event.target.value;
    onModelChange(value === "" ? null : (value as ModelName));
  };

  return (
    <FormControl fullWidth size="small" sx={{ minWidth: 200 }}>
      <InputLabel>Model</InputLabel>
      <Select value={selectedModel || ""} label="Model" onChange={handleChange}>
        <MenuItem value="">
          <em>Default</em>
        </MenuItem>
        {models.map((model) => (
          <MenuItem key={model} value={model}>
            {model}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}


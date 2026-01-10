import { FormControl, InputLabel, Select, MenuItem, type SelectChangeEvent } from "@mui/material";
import type { AgentInfo } from "../../api/types";

interface AgentSelectorProps {
  agents: AgentInfo[];
  selectedAgent: string;
  onAgentChange: (agentId: string) => void;
}

export function AgentSelector({ agents, selectedAgent, onAgentChange }: AgentSelectorProps) {
  const handleChange = (event: SelectChangeEvent<string>) => {
    onAgentChange(event.target.value);
  };

  return (
    <FormControl fullWidth size="small" sx={{ minWidth: 200 }}>
      <InputLabel>Agent</InputLabel>
      <Select value={selectedAgent} label="Agent" onChange={handleChange}>
        {agents.map((agent) => (
          <MenuItem key={agent.key} value={agent.key}>
            {agent.key} - {agent.description}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}


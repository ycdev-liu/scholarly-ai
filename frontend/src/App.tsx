import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./components/Layout/ThemeProvider";
import { AppLayout } from "./components/Layout/AppLayout";
import { ChatWindow } from "./components/Chat/ChatWindow";
import { PaperSearch } from "./components/Paper/PaperSearch";
import { VectorDBManagement } from "./components/VectorDB/VectorDBManagement";
import { Settings } from "./components/Settings/Settings";

function App() {
  return (
    <ThemeProvider mode="light">
      <BrowserRouter>
        <AppLayout>
          <Routes>
            <Route path="/" element={<ChatWindow />} />
            <Route path="/papers" element={<PaperSearch />} />
            <Route path="/vector-db" element={<VectorDBManagement />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </AppLayout>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;

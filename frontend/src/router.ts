import { createRouter, createWebHistory } from 'vue-router'
import ResearchView from './views/ResearchView.vue'
import LibraryView from './views/LibraryView.vue'

// 只有两个工作区路由；服务端/Nginx 会将刷新后的深层路径回退到 index.html。
export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'research', component: ResearchView },
    { path: '/library', name: 'library', component: LibraryView },
  ],
})

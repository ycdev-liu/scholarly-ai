import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import './style.css'

// 全局注入 Pinia 和路由后挂载应用，页面状态不依赖组件间逐层传参。
createApp(App).use(createPinia()).use(router).mount('#app')

<script setup>
import { RouterView, useRoute } from 'vue-router'
import { computed, onMounted } from 'vue'
import MainLayout from './layouts/MainLayout.vue';
import AuthLayout from './layouts/AuthLayout.vue';

const route = useRoute()

onMounted(() => {
  setTimeout(() => {
    window.HSStaticMethods.autoInit()
  }, 100);
})

const layouts = {
  auth: AuthLayout,
  main: MainLayout,
}

const layoutComponent = computed(() => {
  const layoutName = route.meta?.layout || 'main'
  return layouts[layoutName] || layouts.main
})
</script>

<template>
  <component :is="layoutComponent">
    <RouterView />
  </component>
</template>

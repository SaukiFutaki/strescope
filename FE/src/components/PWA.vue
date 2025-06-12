<template>
  <div v-if="showInstallPrompt" class="install-pwa">
    <div class="install-banner">
      <div class="install-content">
        <h3>Install App</h3>
        <p>Install aplikasi ini untuk pengalaman yang lebih baik!</p>
      </div>
      <div class="install-actions">
        <button @click="installPWA" class="install-btn">Install</button>
        <button @click="dismissPrompt" class="dismiss-btn">×</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const name = 'PWA'

const showInstallPrompt = ref(false)

const installPWA = () => {
  showInstallPrompt.value = false
  window.deferredPrompt.prompt()
}

const dismissPrompt = () => {
  showInstallPrompt.value = false
  window.deferredPrompt = null
}

onMounted(() => {
  window.addEventListener('beforeinstallprompt', (event) => {
    event.preventDefault()
    showInstallPrompt.value = true
    window.deferredPrompt = event
  })
})
</script>

<style scoped>
.install-pwa {
  position: fixed;
  bottom: 20px;
  left: 20px;
  right: 20px;
  z-index: 1000;
}

.install-banner {
  background: #42b883;
  color: white;
  padding: 16px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.install-content h3 {
  margin: 0 0 4px 0;
  font-size: 16px;
}

.install-content p {
  margin: 0;
  font-size: 14px;
  opacity: 0.9;
}

.install-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}

.install-btn {
  background: white;
  color: #42b883;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.install-btn:hover {
  background: #f0f0f0;
}

.dismiss-btn {
  background: none;
  border: none;
  color: white;
  font-size: 24px;
  cursor: pointer;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
}

@media (max-width: 640px) {
  .install-banner {
    flex-direction: column;
    text-align: center;
    gap: 12px;
  }
  
  .install-actions {
    justify-content: center;
  }
}
</style>
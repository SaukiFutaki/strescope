<script setup>
import { onMounted, ref } from 'vue';
import { FormWizard, TabContent } from 'vue3-form-wizard'
import 'vue3-form-wizard/dist/style.css'
import Loading from '@/components/Loading.vue'
import { useRouter } from 'vue-router';

const router = useRouter();

const isLoading = ref(true)

const daftarPertanyaan = ref([])

onMounted(async () => {
  console.log('cek stres mounted')

  daftarPertanyaan.value = [
    {
      id: 1,
      pertanyaan: 'Pertanyaan pertama dari BE (?)',
    },
    {
      id: 2,
      pertanyaan: 'Pertanyaan ke-2 dari BE (?)',
    }
  ]

  // TODO:
  // fetch api data pertanyaan kuesioner
  // await fetch
  // update daftarPertanyaan dan isLoading
  isLoading.value = false
})

const validateDataDiri = async () => {
  console.log('validate step 1')

  return true
}

const validateDataKesehatan = async () => {
  console.log('validate step 2')

  return true
}

const validateDataAktivitas = async () => {
  console.log('validate step 3')

  return true
}

const onComplete = () => {
  // kirim data ke api
  alert("Yay. Done!");
  router.push({ name: 'hasil' })

}
</script>

<template>
  <template v-if="isLoading">
    <div class="flex items-center justify-center h-100">
      <loading size="8" />
    </div>
  </template>
  <template v-else>
    <form-wizard @on-complete="onComplete" step-size="xs" color="#3498db">
      <div class="mb-5 text-lg lg:w-160">
        Mari mencari tahu tingkat stress kamu dengan menjawab pertanyaan dibawah ini sobat StressCope!
      </div>

      <tab-content title="Data Diri" :before-change="validateDataDiri">
        <div class="card">
          <div class="mb-3">
            <label for="usiaInput" class="block text-sm font-medium mb-2">Usia</label>
            <input type="text" id="usiaInput" name="usia"
              class="py-2.5 sm:py-3 px-4 block w-full border-gray-200 rounded-lg sm:text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none"
              required>
          </div>
          <div class="mb-3">
            <label for="jkSelect" class="block text-sm font-medium mb-2">Jenis Kelamin</label>
            <select id="jkSelect" name="jk"
              class="py-3 px-4 pe-9 block w-full border-gray-200 rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none dark:bg-neutral-900 dark:border-neutral-700 dark:text-neutral-400 dark:placeholder-neutral-500 dark:focus:ring-neutral-600"
              required>
              <option value="" selected hidden>Pilih</option>
              <option value="Laki-laki">Laki-laki</option>
              <option value="Perempuan">Perempuan</option>
            </select>
          </div>
          <div class="mb-3">
            <label for="pekerjaanInput" class="block text-sm font-medium mb-2">Pekerjaan</label>
            <input type="text" id="pekerjaanInput" name="pekerjaan"
              class="py-2.5 sm:py-3 px-4 block w-full border-gray-200 rounded-lg sm:text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none"
              required>
          </div>
        </div>
      </tab-content>
      <tab-content title="Data Kesehatan" :before-change="validateDataKesehatan">
        <div class="card">
          <div class="mb-3">
            <label for="durasiTidurInput" class="block text-sm font-medium mb-2">Durasi Tidur (Jam)</label>
            <input type="text" id="durasiTidurInput" name="durasi_tidur"
              class="py-2.5 sm:py-3 px-4 block w-full border-gray-200 rounded-lg sm:text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none"
              required>
          </div>
          <div class="mb-3">
            <label for="kualitasTidurSelect" class="block text-sm font-medium mb-2">Kualitas Tidur</label>
            <select id="kualitasTidurSelect" name="kualitas_tidur"
              class="py-3 px-4 pe-9 block w-full border-gray-200 rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none dark:bg-neutral-900 dark:border-neutral-700 dark:text-neutral-400 dark:placeholder-neutral-500 dark:focus:ring-neutral-600"
              required>
              <option value="" selected hidden>Pilih</option>
              <template v-for="i in 10">
                <option :value="i">{{ i }}</option>
              </template>
            </select>
          </div>
          <div class="mb-3">
            <label for="detakJantungInput" class="block text-sm font-medium mb-2">Detak Jantung (BPM)</label>
            <input type="text" id="detakJantungInput" name="bpm"
              class="py-2.5 sm:py-3 px-4 block w-full border-gray-200 rounded-lg sm:text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none"
              required>
          </div>
          <div class="mb-3">
            <label for="kategoriBMISelect" class="block text-sm font-medium mb-2">Kategori BMI</label>
            <select id="kategoriBMISelect" name="kategori_bmi"
              class="py-3 px-4 pe-9 block w-full border-gray-200 rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none dark:bg-neutral-900 dark:border-neutral-700 dark:text-neutral-400 dark:placeholder-neutral-500 dark:focus:ring-neutral-600"
              required>
              <option value="" selected hidden>Pilih</option>
              <option value="underweight">Underweight</option>
              <option value="normal">Normal</option>
              <option value="overweight">Overweight</option>
              <option value="obese">Obese</option>
            </select>
          </div>
          <div class="mb-3">
            <label for="tekananDarahInput" class="block text-sm font-medium mb-2">Tekanan Darah (Opsional)</label>
            <input type="text" id="tekananDarahInput" name="tekanan_darah"
              class="py-2.5 sm:py-3 px-4 block w-full border-gray-200 rounded-lg sm:text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none">
          </div>
        </div>
      </tab-content>
      <tab-content title="Data Aktivitas" :before-change="validateDataAktivitas">
        <div class="card">
          <div class="mb-3">
            <label for="levelAktivitasFisikInput" class="block text-sm font-medium mb-2">Level Aktivitias Fisik (1-1000)</label>
            <input type="text" id="levelAktivitasFisikInput" name="level_aktivitias_fisik"
              class="py-2.5 sm:py-3 px-4 block w-full border-gray-200 rounded-lg sm:text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none"
              required>
          </div>
          <div class="mb-3">
            <label for="langkahHarianInput" class="block text-sm font-medium mb-2">Jumlah Langkah Harian (Langkah)</label>
            <input type="text" id="langkahHarianInput" name="langkah_harian"
              class="py-2.5 sm:py-3 px-4 block w-full border-gray-200 rounded-lg sm:text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none"
              required>
          </div>
          <div class="mb-3">
            <label for="gangguanTidurSelect" class="block text-sm font-medium mb-2">Memiliki Gangguan Tidur</label>
            <select id="gangguanTidurSelect" name="gangguan_tidur"
              class="py-3 px-4 pe-9 block w-full border-gray-200 rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none dark:bg-neutral-900 dark:border-neutral-700 dark:text-neutral-400 dark:placeholder-neutral-500 dark:focus:ring-neutral-600"
              required>
              <option value="" selected hidden>Pilih</option>
              <option value="Iya">Iya</option>
              <option value="Tidak">Tidak</option>
            </select>
          </div>
          <!-- level aktivitas fisik, jumlah langkah harian, memiliki gangguan tidur (optional) -->
        </div>
      </tab-content>
      <tab-content title="Kuesioner">
        <div class="card">
          <div class="mb-4">
            <h3 class="text-lg font-medium">Pilihlah jawaban yang paling sesuai dengan kamu</h3>
            <p>1 = Tidak berlaku bagi saya sama sekali</p>
            <p>2 = Berlaku untuk saya sampai pada tingkat tertentu, atau sebagian waktu.</p>
            <p>3 = Diterapkan pada saya hingga tingkat yang cukup besar, atau sebagian besar.</p>
            <p>4 = Diterapkan pada saya hingga tingkat yang sangat besar</p>
          </div>

          <template v-for="pertanyaan in daftarPertanyaan">
            <div class="mb-3">
              <label :for="`pertanyaan-${pertanyaan.id}`" class="block font-medium mb-2">
                {{ pertanyaan.pertanyaan }}
              </label>
              <div class="flex gap-x-6" :id="`pertanyaan-${pertanyaan.id}`">
                <template v-for="i in 4">
                  <div class="flex">
                    <input type="radio" :name="`jawaban[${pertanyaan.id}]`" :value="i"
                      class="shrink-0 mt-0.5 border-gray-200 rounded-full text-blue-600 focus:ring-blue-500 checked:border-blue-500 disabled:opacity-50 disabled:pointer-events-none"
                      :id="`jawaban-${pertanyaan.id}-${i}`">
                    <label :for="`jawaban-${pertanyaan.id}-${i}`" class="text-sm text-gray-500 ms-2">{{ i }}</label>
                  </div>
                </template>
              </div>
            </div>
          </template>
          <template v-if="daftarPertanyaan.length == 0">
            <p class="text-center py-7">Tidak ada pertanyaan</p>
          </template>
        </div>
      </tab-content>
    </form-wizard>
  </template>
</template>

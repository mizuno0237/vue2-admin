<template>
<<<<<<< HEAD
    <div id="app">
        <el-button @click="visible = true">Button</el-button>
        <el-dialog :visible.sync="visible" title="Hello world">
            <p>Try Element</p>
        </el-dialog>
    </div>
</template>

<script>
    export default {
         data: function() {
                return { visible: false }
        }
    }
</script>

<style>

</style>
=======
  <div id="app">
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>
>>>>>>> 95c7bf5d10d5071136ed34cdd8ba669a281de523

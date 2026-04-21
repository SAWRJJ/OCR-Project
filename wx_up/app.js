App({
  onLaunch: function () {
    console.log('小程序启动')
  },
  globalData: {
    userInfo: null,
    uploadUrl: 'http://192.168.4.126:5000/upload' // 请替换为实际的后端上传地址
  }
})
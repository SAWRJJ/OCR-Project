const app = getApp()

Page({
  data: {
    tempFilePath: '',
    imageInfo: '',
    uploadResult: ''
  },

  // 选择图片
  chooseImage() {
    const that = this
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['album', 'camera'],
      success(res) {
        const tempFilePath = res.tempFilePaths[0]
        that.setData({
          tempFilePath: tempFilePath,
          uploadResult: ''
        })
        
        // 获取图片信息
        wx.getImageInfo({
          src: tempFilePath,
          success(imageRes) {
            const info = `尺寸: ${imageRes.width}x${imageRes.height}px | 大小: ${(res.tempFiles[0].size / 1024).toFixed(1)}KB`
            that.setData({
              imageInfo: info
            })
          }
        })
      },
      fail(err) {
        wx.showToast({
          title: '选择图片失败',
          icon: 'none'
        })
        console.error('选择图片失败:', err)
      }
    })
  },

  // 上传图片到服务器
  uploadImage() {
    if (!this.data.tempFilePath) {
      wx.showToast({
        title: '请先选择图片',
        icon: 'none'
      })
      return
    }

    const that = this
    wx.showLoading({
      title: '上传中...'
    })

    wx.uploadFile({
      url: app.globalData.uploadUrl, // 替换为实际的后端上传接口
      filePath: this.data.tempFilePath,
      name: 'file', // 后端接收文件的字段名
      formData: {
        'user': '微信用户',
        'timestamp': Date.now().toString()
      },
      success(res) {
        const data = JSON.parse(res.data)
        wx.hideLoading()
        
        if (res.statusCode === 200 && data.success) {
          wx.showToast({
            title: '上传成功',
            icon: 'success'
          })
          that.setData({
            uploadResult: JSON.stringify(data, null, 2)
          })
        } else {
          wx.showToast({
            title: '上传失败',
            icon: 'none'
          })
          that.setData({
            uploadResult: `上传失败: ${data.message || '未知错误'}`
          })
        }
      },
      fail(err) {
        wx.hideLoading()
        wx.showToast({
          title: '网络错误',
          icon: 'none'
        })
        that.setData({
          uploadResult: `上传失败: ${err.errMsg}`
        })
        console.error('上传失败:', err)
      }
    })
  }
})
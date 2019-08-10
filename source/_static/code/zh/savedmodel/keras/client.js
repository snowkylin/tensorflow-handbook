const Jimp = require('jimp')
const superagent = require('superagent')

const url = 'http://localhost:8501/v1/models/MLP:predict'

const getPixelGrey = (pic, x, y) => {
  const pointColor = pic.getPixelColor(x, y)
  const { r, g, b } = Jimp.intToRGBA(pointColor)
  const gray =  +(r * 0.299 + g * 0.587 + b * 0.114).toFixed(0)
  return [ gray / 255 ]
}

const getPicGreyArray = async (fileName) => {
  const pic = await Jimp.read(fileName)
  const resizedPic = pic.resize(28, 28)
  const greyArray = []
  for ( let i = 0; i< 28; i ++ ) {
    let line = []
    for (let j = 0; j < 28; j ++) {
      line.push(getPixelGrey(resizedPic, j, i))
    }
    console.log(line.map(_ => _ > 0.3 ? ' ' : '1').join(' '))
    greyArray.push(line)
  }
  return greyArray
}

const evaluatePic = async (fileName) => {
  const arr = await getPicGreyArray(fileName)
  const result = await superagent.post(url)
    .send({
      instances: [arr]
    })
  result.body.predictions.map(res => {
    const sortedRes = res.map((_, i) => [_, i])
    .sort((a, b) => b[0] - a[0])
    console.log(`我们猜这个数字是${sortedRes[0][1]}，概率是${sortedRes[0][0]}`)
  })
}

evaluatePic('test_pic_tag_5.png')
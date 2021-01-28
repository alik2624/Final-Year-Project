document.addEventListener('DOMContentLoaded', () => {
//The First Phase
const image = document.getElementById('tron')


//The Second Phase
Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceExpressionNet.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
  ]).then(drawboxes)
  
  function startVideo() {
    drawboxes()
  }

  // The Third Phase
// startVideo()
  async function drawboxes() {
    
    //create the canvas from video element as we have created above
    const canvas = faceapi.createCanvasFromMedia(image);
    //append canvas to body or the dom element where you want to append it
    document.body.append(canvas)
    // displaySize will help us to match the dimension with video screen and accordingly it will draw our detections
    // on the streaming video screen
    const displaySize = { width: image.width, height: image.height }
    faceapi.matchDimensions(canvas, displaySize)
    
      const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceExpressions().withFaceDescriptors()
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      const resizedDetections = faceapi.resizeResults(detections, displaySize)
      
      
      faceapi.draw.drawDetections(canvas, resizedDetections)
      faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
      faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
      
      //The Labelled Additional Code
      const labels = ["jeff",'olivia','Garett']
      var t0 = performance.now() 
      const labeledFaceDescriptors = await Promise.all(
          labels.map(async label => {
              // fetch image data from urls and convert blob to HTMLImage element
              const img = document.getElementById(`${label}`) 
              
              
              // detect the face with the highest score in the image and compute it's landmarks and face descriptor
              const fullFaceDescription = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
              
              if (!fullFaceDescription) {
              throw new Error(`no faces detected for ${label}`)
              }
              
              const faceDescriptors = [fullFaceDescription.descriptor]
              return new faceapi.LabeledFaceDescriptors(label, faceDescriptors)
          })
      );
      var t1 = performance.now()

      console.log("The Array function took" + (t1-t0) + "milliseconds")
      

      const maxDescriptorDistance = 0.6
      const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance)

      const results = resizedDetections.map(fd => faceMatcher.findBestMatch(fd.descriptor))

      results.forEach((bestMatch, i) => {
          const box = resizedDetections[i].detection.box
          const text = bestMatch.toString()
          const drawBox = new faceapi.draw.DrawBox(box, { label: text })
          drawBox.draw(canvas)
      })

    
  }

})
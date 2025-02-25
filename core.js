/* eslint-disable */
import * as tf from "@tensorflow/tfjs";
import "babel-polyfill";
import links from "./links";
tf.ENV.set("WEBGL_PACK", false); // This needs to be done otherwise things run very slow v1.0.4

class Core {
	constructor() {
		Promise.all([this.loadMobileNetStyleModel(), this.loadSeparableTransformerModel()]).then(
			([styleNet, transformNet]) => {
				console.log("Loaded styleNet");
				this.styleNet = styleNet;
				this.transformNet = transformNet;
			}
		);

		//Checks if bundle is part of CSnap
		this.ide = null;
		if (typeof world !== "undefined") {
			this.ide = world.children[0];
		}
	}

	generateStylizedImage(options) {
		let generic = {
			contentImage: "images/beach.jpg",
			sourceImage: "images/statue_of_liberty.jpg",
			styleModel: "mobilenet",
			transformModel: "separable",
			styleRatio: 0.5,
			contentSize: 100,
			sourceSize: 100,
			download: true,
		};

		if (options) {
			Object.assign(generic, options);
		}

		this.contentImg = document.getElementById("base-img");
		this.styleImg = document.getElementById("style-img");
		this.stylized = document.getElementById("style-canvas");

		// Get width and height of base image
		const baseWidth = this.contentImg.width;
		const baseHeight = this.contentImg.height;

		this.connectImageAndSizeScale(this.contentImg, generic.contentSize);
		this.connectImageAndSizeScale(this.styleImg, generic.sourceSize);

		this.styleRatio = generic.styleRatio;

		// Calls the block that loads the progress bar to user
		if (typeof world !== "undefined") {
			let ide = world.children[0];
			ide.broadcast("startProgress");
		}

		Promise.all([this.loadStyleModel(generic.styleModel), this.loadTransformModel(generic.transformModel)]).then(
			([styleNet, transformNet]) => {
				console.log("Loaded styleNet");
				this.styleNet = styleNet;
				this.transformNet = transformNet;

				this.startStyling().finally(() => {
					let a = document.createElement("a");

					// document.querySelector("#converted-image").src = this.stylized.toDataURL("image/png", 1.0);
					const ctx = this.stylized.getContext("2d");
					const img = new Image();
					img.src = this.stylized.toDataURL("image/png", 1.0); // Replace with your image URL
					img.onload = () => {
						// Draw the image on the top-left corner of the canvas
						ctx.drawImage(img, 0, 0);

						// Create a new canvas for the cropped image
						const croppedCanvas = document.createElement("canvas");
						croppedCanvas.width = baseWidth; // Set to the cropped width
						croppedCanvas.height = baseHeight; // Set to the cropped height
						const croppedCtx = croppedCanvas.getContext("2d");

						// Extract the relevant portion from the original canvas
						const imageData = ctx.getImageData(0, 0, baseWidth, baseHeight);
						croppedCtx.putImageData(imageData, 0, 0);

						// The croppedCanvas now contains the cropped image
						document.querySelector("#converted-image").src = croppedCanvas.toDataURL("image/png", 1.0);
						// document.body.appendChild(croppedCanvas); // Append to the document to see the result
					};

					if (generic.download) {
						a.setAttribute("download", "output.png");
						a.setAttribute("href", this.stylized.toDataURL("image/png", 1.0));
						document.body.appendChild(a);
						a.click();
						document.body.removeChild(a);
					}
					console.log("Finished");
					// Calls the block that hides the progress bar to user
					if (typeof world !== "undefined") {
						let ide = world.children[0];
						ide.broadcast("endProgress");
					}
				});
			}
		);
	}

	connectImageAndSizeScale(img, scaleRatio) {
		img.height = img.height * scaleRatio;
		img.width = img.width * scaleRatio;
		if (img.style.width) {
			// If this branch is triggered, then that means the image was forced to a square using
			// a fixed pixel value.
			img.style.width = img.height + "px"; // Fix width back to a square
		}
	}

	async loadStyleModel(style) {
		return style == "inception" ? this.loadInceptionStyleModel() : this.loadMobileNetStyleModel();
	}

	async loadTransformModel(transform) {
		return transform == "original" ? this.loadOriginalTransformerModel() : this.loadSeparableTransformerModel();
	}

	async loadMobileNetStyleModel() {
		if (!this.mobileStyleNet) {
			this.mobileStyleNet = await tf.loadGraphModel("/csnap_pro/csdt/features/nst/lib/saved_model_style_js/model.json");
			if (this.ide) {
				this.ide.broadcast("fastModelLoad");
			}
		}

		return this.mobileStyleNet;
	}

	async loadInceptionStyleModel() {
		if (!this.inceptionStyleNet) {
			this.inceptionStyleNet = await tf.loadGraphModel(
				"/csnap_pro/csdt/features/nst/lib/saved_model_style_inception_js/model.json"
			);
			if (this.ide) {
				this.ide.broadcast("highModelLoad");
			}
		}

		return this.inceptionStyleNet;
	}

	async loadOriginalTransformerModel() {
		if (!this.originalTransformNet) {
			this.originalTransformNet = await tf.loadGraphModel(
				"/csnap_pro/csdt/features/nst/lib/saved_model_transformer_js/model.json"
			);
			if (this.ide) {
				this.ide.broadcast("highTransformLoad");
			}
		}

		return this.originalTransformNet;
	}

	async loadSeparableTransformerModel() {
		if (!this.separableTransformNet) {
			this.separableTransformNet = await tf.loadGraphModel(
				"/csnap_pro/csdt/features/nst/lib/saved_model_transformer_separable_js/model.json"
			);
			if (this.ide) {
				this.ide.broadcast("fastTransformLoad");
			}
		}

		return this.separableTransformNet;
	}

	async startStyling() {
		await tf.nextFrame();
		console.log("Generating 100D style representation");
		await tf.nextFrame();
		let bottleneck = await tf.tidy(() => {
			return this.styleNet.predict(tf.browser.fromPixels(this.styleImg).toFloat().div(tf.scalar(255)).expandDims());
		});
		if (this.styleRatio !== 1.0) {
			console.log("Generating 100D identity style representation");
			await tf.nextFrame();
			const identityBottleneck = await tf.tidy(() => {
				return this.styleNet.predict(tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims());
			});
			const styleBottleneck = bottleneck;
			bottleneck = await tf.tidy(() => {
				const styleBottleneckScaled = styleBottleneck.mul(tf.scalar(this.styleRatio));
				const identityBottleneckScaled = identityBottleneck.mul(tf.scalar(1.0 - this.styleRatio));
				return styleBottleneckScaled.addStrict(identityBottleneckScaled);
			});
			styleBottleneck.dispose();
			identityBottleneck.dispose();
		}
		console.log("Stylizing image...");
		await tf.nextFrame();
		const stylized = await tf.tidy(() => {
			return this.transformNet
				.predict([tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims(), bottleneck])
				.squeeze();
		});
		await tf.browser.toPixels(stylized, this.stylized);
		bottleneck.dispose(); // Might wanna keep this around
		stylized.dispose();
	}

	async startCombining() {
		await tf.nextFrame();
		// this.combineButton.textContent =
		//   "Generating 100D style representation of image 1";
		await tf.nextFrame();
		const bottleneck1 = await tf.tidy(() => {
			return this.styleNet.predict(
				tf.browser.fromPixels(this.combStyleImg1).toFloat().div(tf.scalar(255)).expandDims()
			);
		});

		// this.combineButton.textContent =
		//   "Generating 100D style representation of image 2";
		await tf.nextFrame();
		const bottleneck2 = await tf.tidy(() => {
			return this.styleNet.predict(
				tf.browser.fromPixels(this.combStyleImg2).toFloat().div(tf.scalar(255)).expandDims()
			);
		});

		// this.combineButton.textContent = "Stylizing image...";
		await tf.nextFrame();
		const combinedBottleneck = await tf.tidy(() => {
			const scaledBottleneck1 = bottleneck1.mul(tf.scalar(1 - this.combStyleRatio));
			const scaledBottleneck2 = bottleneck2.mul(tf.scalar(this.combStyleRatio));
			return scaledBottleneck1.addStrict(scaledBottleneck2);
		});

		const stylized = await tf.tidy(() => {
			return this.transformNet
				.predict([
					tf.browser.fromPixels(this.combContentImg).toFloat().div(tf.scalar(255)).expandDims(),
					combinedBottleneck,
				])
				.squeeze();
		});
		await tf.browser.toPixels(stylized, this.combStylized);
		bottleneck1.dispose(); // Might wanna keep this around
		bottleneck2.dispose();
		combinedBottleneck.dispose();
		stylized.dispose();
	}

	async benchmark() {
		const x = tf.randomNormal([1, 256, 256, 3]);
		const bottleneck = tf.randomNormal([1, 1, 1, 100]);

		let styleNet = await this.loadInceptionStyleModel();
		let time = await this.benchmarkStyle(x, styleNet);
		styleNet.dispose();

		styleNet = await this.loadMobileNetStyleModel();
		time = await this.benchmarkStyle(x, styleNet);
		styleNet.dispose();

		let transformNet = await this.loadOriginalTransformerModel();
		time = await this.benchmarkTransform(x, bottleneck, transformNet);
		transformNet.dispose();

		transformNet = await this.loadSeparableTransformerModel();
		time = await this.benchmarkTransform(x, bottleneck, transformNet);
		transformNet.dispose();

		x.dispose();
		bottleneck.dispose();
	}

	async benchmarkStyle(x, styleNet) {
		const profile = await tf.profile(() => {
			tf.tidy(() => {
				const dummyOut = styleNet.predict(x);
				dummyOut.print();
			});
		});
		console.log(profile);
		const time = await tf.time(() => {
			tf.tidy(() => {
				for (let i = 0; i < 10; i++) {
					const y = styleNet.predict(x);
					y.print();
				}
			});
		});
		console.log(time);
	}

	async benchmarkTransform(x, bottleneck, transformNet) {
		const profile = await tf.profile(() => {
			tf.tidy(() => {
				const dummyOut = transformNet.predict([x, bottleneck]);
				dummyOut.print();
			});
		});
		console.log(profile);
		const time = await tf.time(() => {
			tf.tidy(() => {
				for (let i = 0; i < 10; i++) {
					const y = transformNet.predict([x, bottleneck]);
					y.print();
				}
			});
		});
		console.log(time);
	}
}

// function validateTextureSize(width, height) {
//   var maxTextureSize = environment_1.ENV.get('WEBGL_MAX_TEXTURE_SIZE');
//   if ((width <= 0) || (height <= 0)) {
//       var requested = "[" + width + "x" + height + "]";
//       if(typeof world !== 'undefined'){
//           let ide = world.children[0]
//           ide.broadcast('sizeError')
//       }
//       throw new Error('Requested texture size ' + requested + ' is invalid.');
//   }
//   if ((width > maxTextureSize) || (height > maxTextureSize)) {
//       var requested = "[" + width + "x" + height + "]";
//       var max = "[" + maxTextureSize + "x" + maxTextureSize + "]";
//       if(typeof world !== 'undefined'){
//           let ide = world.children[0]
//           ide.broadcast('sizeError')
//       }
//       throw new Error('Requested texture size ' + requested +
//           ' greater than WebGL maximum on this browser / GPU ' + max + '.');
//   }
// }

window.application = new Core();

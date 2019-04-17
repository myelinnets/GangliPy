# GangliPy

GangliPy is a collection of neural networks evaluated to be the best for certain types of tasks.

# In Progress


## Auto OCR

The first project here is an attempt to recognize all 60,000 unicode characters using a single hidden layer.

Fully connected input and output layers have proven sufficient for recognizing 16x16 unicode images, and can even 
predict never before seen characters.

![OCR example](https://i.imgur.com/2mwf7XQ.jpg)

However, 'predicting never before seen characters' fades to noise when enough characters are seen, so another layer is 
 needed to make this 'variational' and keep realistic guesses when failing to perfectly predict.

## Unicode Regognition Tests

The unicode recognition tests I'm using for Auto OCR will need to be pulled out and placed into a seperate testing and 
evaluation repository.

# Installation

There is currently no pip repository available. Once Auto OCR is finished, it will be pulled out into its own repo and 
will be added as a requirement to this one.

#License

GangliPy is distributed under the terms of 
[GNU AGPL V3.0](https://choosealicense.com/licenses/agpl-3.0)

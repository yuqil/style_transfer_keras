# -*- coding: utf-8 -*-
import urllib2
import urllib
import json
import time
import base64
from PIL import Image

http_url = 'https://api-us.faceplusplus.com/humanbodypp/beta/segment'
key = "dtcZdNW5mv7VY8szko9xq_Lvkyk9UEwi"
secret = "iiSAGn7-8TybE6_ALLQmCRJY0IDUvmpW"

# img = Image.open('img.jpeg')
# img = img.resize((512, 512),Image.BILINEAR)
# img.save('img2.jpeg', 'jpeg')

def get_confidence(filepath):
	boundary = '----------%s' % hex(int(time.time() * 1000))
	data = []
	data.append('--%s' % boundary)
	data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
	data.append(key)
	data.append('--%s' % boundary)
	data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
	data.append(secret)
	data.append('--%s' % boundary)
	fr = open(filepath, 'rb')
	data.append('Content-Disposition: form-data; name="%s"; filename="co33.jpg"' % 'image_file')
	data.append('Content-Type: %s\r\n' % 'application/octet-stream')
	data.append(fr.read())
	fr.close()
	data.append('--%s--\r\n' % boundary)

	http_body = '\r\n'.join(data)
	# buld http request
	req = urllib2.Request(http_url)
	# header
	req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
	req.add_data(http_body)
	try:
		# post data to server
		resp = urllib2.urlopen(req, timeout=5)
		# get response
		qrcont = resp.read()
		data = json.loads(qrcont)

		imgdata = base64.b64decode(data['result'])
		filename = 'confi_map.jpg'  # I assume you have a way of picking unique filenames
		with open(filename, 'wb') as f:
			f.write(imgdata)

		return filename

	except urllib2.HTTPError as e:
		print(e.read())

def replace_body(orig_path, trans_path, dest_path):
	threshold = 150
	confi_map_path = get_confidence(orig_path)

	confi_map = Image.open(confi_map_path)
	orig_img  = Image.open(orig_path)
	trans_img = Image.open(trans_path)

	width, height = confi_map.size
	for x in range(width):
		for y in range(height):
			current_color = confi_map.getpixel((x, y))
			if current_color >= threshold:
				trans_img.putpixel((x, y), orig_img.getpixel((x, y)))
	trans_img.save(dest_path, 'jpeg')

if __name__ == '__main__':
	orig_path = 'orig.jpeg'
	trans_path = 'trans.png'
	new_path = 'new.jpeg'
	replace_body(orig_path, trans_path, new_path)






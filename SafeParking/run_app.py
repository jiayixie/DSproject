#!/usr/bin/env python
from app import app
#app.run(debug = True)
app.run(host='0.0.0.0', port=5000, debug=True)

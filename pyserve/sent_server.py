import logging
import falcon
from falcon_cors import CORS
import json
import waitress
from PUBW_BiLSTM_BN import PUBW_BiLSTM_BN

hostname = '192.168.4.250'
logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
l = logging.getLogger()
cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['http://192.168.4.250:8081'],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )

class SentResource:


    bilstm=None


    def __init__(self):
        l.info("create pub-bilstm-bn:")
        self.bilstm = PUBW_BiLSTM_BN()
        l.info("load keras model:")
        self.bilstm.loadKeras()
        l.info("keras model loaded.")
        segs = self.bilstm.opinion(["哈哈哈。", "那酒店也就呵呵了。","到半夜竟然没暖气,怎么住啊????!!!!!!!!!!"])
        l.info("inference done.")
        print(segs)

    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.set_header('Access-Control-Allow-Origin', 'http://%s:8081'%hostname)
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        sentence = req.get_param('q', True)
        l.info('sentence:', sentence)
        words = self.bilstm.cut([sentence])
        l.info("seg result:", words)
        print("ALL-DONE")
        resp.media = {"words":words}


    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', 'http://%s:8081'%hostname)
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        data = req.stream.read(req.content_length)
        reqdata = json.loads(data, encoding='utf-8')
        print('sentence:', reqdata['sents'])
        sentences = reqdata['sents']
        sentences = [s.strip() for s in sentences if len(s.strip())>0]
        if not isinstance(sentences, list):
            sentences = [sentences]
        opinions = self.bilstm.opinion(sentences)
        print("opinion result:", opinions)
        resp.media = {'data':{"opn":opinions}}

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/opinion', SentResource())
    waitress.serve(api, port=8090, url_scheme='http')
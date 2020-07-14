import logging
import falcon
from falcon_cors import CORS
import waitress
from seg_server import SegResource #BiLSTM and CRF
from sent_server import SentResource

hostname = '192.168.4.250'
logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
l = logging.getLogger()
cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['http://%s:8081'%hostname],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/segment', SegResource())
    api.add_route('/opinion', SentResource())
    waitress.serve(api, port=8090, url_scheme='http')
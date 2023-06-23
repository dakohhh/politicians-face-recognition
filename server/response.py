from flask import jsonify, Response




def CustomResponse(message, status=200, success=True, data=None, *args , **kwargs) -> Response:


    response = {
        "data" : data,
        "status" : status,
        "message": message,
        "success": success if status not in (300, 400, 500) else False,
    }


    return jsonify(response, **kwargs), status
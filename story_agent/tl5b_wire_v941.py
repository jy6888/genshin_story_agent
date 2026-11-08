"""One implementation self-repair: cumulative file numbering, new-only billing."""
from story_agent.tl5b_wire import AuditedDeepSeek as Previous, accounted, delta


class AuditedDeepSeek(Previous):
    _numbering_request = False

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.run_id='TL5b:v0.94.1:'+self.out.name+':'+self.topic

    def gate(self):
        value = super().gate()
        if self._numbering_request:
            value = dict(value, attempts=value['attempts']+self.baseline['attempts'])
        return value

    def _request(self, request):
        # The frozen transport uses gate()['attempts'] for the next evidence ID.
        # Scope the offset to its request lifecycle, retaining new-only public usage.
        self._numbering_request = True
        try: return super()._request(request)
        finally: self._numbering_request = False

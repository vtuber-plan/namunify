// Sample obfuscated JavaScript for testing
var a = function(b, c) {
    var d = b + c;
    return d * 2;
};

var e = {
    f: function(g) {
        var h = g.split("");
        var i = h.reverse();
        return i.join("");
    },
    j: function(k, l) {
        var m = k * l;
        var n = m / 2;
        return n;
    }
};

function p(q, r) {
    var s = "";
    for (var t = 0; t < q.length; t++) {
        s += q[t] + r;
    }
    return s;
}

// This function has nested scopes
function u(v) {
    var w = v * 2;
    return function(x) {
        var y = x + w;
        return y * 3;
    };
}

// Class with obfuscated names
class z {
    constructor(aa, ab) {
        this.ac = aa;
        this.ad = ab;
    }

    ae() {
        return this.ac + this.ad;
    }

    static af(ag) {
        return new z(ag, ag * 2);
    }
}

// Export
module.exports = {
    a: a,
    e: e,
    p: p,
    u: u,
    z: z
};

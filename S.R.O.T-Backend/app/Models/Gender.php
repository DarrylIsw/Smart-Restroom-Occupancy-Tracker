<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Gender extends Model
{
    use HasFactory;

    protected $fillable = ['label'];

    // Each gender can be associated with multiple toilets
    public function toilets()
    {
        return $this->hasMany(Toilet::class);
    }
}

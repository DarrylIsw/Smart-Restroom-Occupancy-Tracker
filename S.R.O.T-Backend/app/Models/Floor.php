<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Floor extends Model
{
    use HasFactory;

    protected $fillable = ['building_id', 'floor_number'];

    // Each floor belongs to a building
    public function building()
    {
        return $this->belongsTo(Building::class);
    }

    // Each floor can have multiple toilets
    public function toilets()
    {
        return $this->hasMany(Toilet::class);
    }
}
